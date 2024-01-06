#include "llvm/Support/TargetSelect.h"
//#include "llvm/Analysis/TargetTransformInfo.h"
//#include "llvm/IR/Module.h"
//#include "llvm/Passes/OptimizationLevel.h"
//#include "llvm/Passes/PassBuilder.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace legate {
  class Task;
  class Runtime;
  using ArrayID = int64_t;
  static ArrayID idCounter = 0;
}

namespace numpy {
  enum DType {
    Double,
  };

  size_t dtype_size(DType dtype) {
    switch(dtype) {
      case Double:
        return 8;
      default:
        assert(false);
        return 0;
    }
  }

  mlir::Type dtype_to_mlir_type(DType dtype, mlir::MLIRContext* ctx) {
//    mlir::Bol
    switch (dtype) {
      case Double:
        return mlir::Float64Type::get(ctx);
      default:
        assert(false);
        return mlir::Float16Type::get(ctx);
    }
  }

  class Array {
  public:
    Array(DType dtype, const std::vector<int64_t>& dims) : ndim(dims.size()), dims(dims), dtype(dtype), strides(dims.size()) {
      // Store data in a C-order layout.
      int64_t stride = 1;
      for (int i = this->ndim - 1; i >= 0; i--) {
        this->strides[i] = stride;
        stride *= this->dims[i];
      }
      this->data = malloc(stride * dtype_size(dtype));
      this->id = legate::idCounter++;
    }

    ~Array() {
      free(this->data);
    }

    // Delete the copy constructor for arrays to make sure that we don't
    // copy it since we have an explicit allocation inside.
    Array(const Array&) = delete;

    template<typename T, int N>
    StridedMemRefType<T, N> toMemref() {
      assert(N == this->ndim);
      assert(sizeof(T) == dtype_size(this->dtype));

      StridedMemRefType<T, N> memref{};
      memref.basePtr = static_cast<T*>(this->data);
      memref.data = static_cast<T*>(this->data);
      memref.offset = 0;
      for (int i = 0; i < N; i++) {
        memref.sizes[i] = this->dims[i];
        memref.strides[i] = this->strides[i];
      }
      return memref;
    }

    legate::ArrayID getID() {
      return this->id;
    }

    const std::vector<int64_t>& shape() {
      return this->dims;
    }

    const std::vector<int64_t>& stride() {
      return this->strides;
    }

    bool isTemporary() {
      return this->temporary;
    }

    void setTemporary(bool val) {
      this->temporary = val;
    }

  public:
    int32_t ndim;
    DType dtype;
  private:
    void* data;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    legate::ArrayID id;
    bool temporary = false;
  };
}

namespace legate {
  class Task {
  public:
    Task() : arrays() {}
    void addArray(numpy::Array& arr) {
      this->arrays.push_back(arr);
    }
    const std::vector<std::reference_wrapper<numpy::Array>>& getArrays() {
      return this->arrays;
    }
    void setMLIRBody(mlir::OwningOpRef<mlir::ModuleOp> func) {
      this->func = std::move(func);
    }
    mlir::OwningOpRef<mlir::ModuleOp>& getMLIRBody() {
      return this->func;
    }
    void dumpMLIRBody();
  private:
    std::vector<std::reference_wrapper<numpy::Array>> arrays;
    // func is a module with a single function in it.
    mlir::OwningOpRef<mlir::ModuleOp> func;
  };

  class Runtime {
  public:
    Runtime(int windowSize) : windowSize(windowSize) {}

    void flush() {
      if (this->outstanding.size() == 0) { return; }
      // Flush and execute all queued tasks. This will also kick off
      // the fusion and optimization process.
      // std::cout << "Queued tasks" << std::endl;
      // this->dumpQueuedTasks();

      // To test out how some of these optimizations may work, we'll
      // first start by placing all of the task bodies into a single
      // function that accepts an ordered union of all arguments to each
      // task. This first stage will not consider what is possible to fuse,
      // but will fuse everything and then see what optimizations the
      // system can do given the fusion.

      auto mlirBuildStart = std::chrono::high_resolution_clock::now();

      auto ctx = this->mlirContext.get();
      mlir::OpBuilder builder(ctx);
      mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
      builder.setInsertionPointToEnd(module->getBody());
      auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "AAAAA"));

      // Deduplicate the list of arrays.
      std::vector<ArrayID> arrayIDs;
      std::map<ArrayID, std::reference_wrapper<numpy::Array>> idToArray;
      {
        std::set<ArrayID> arrayIDSet;
        for (auto& t : this->outstanding) {
          for (auto a : t.getArrays()) {
            auto id = a.get().getID();
            if (arrayIDSet.find(id) == arrayIDSet.end()) {
              arrayIDs.push_back(id);
              arrayIDSet.insert(id);
              idToArray.emplace(id, a);
            }
          }
        }
      }

      std::vector<mlir::Type> newFuncTypeArgs;
      for (auto id : arrayIDs) {
        // TODO (rohany): Have to make sure to not accidentally
        //  copy the underlying array, leading to a double-free.
        auto& arr = idToArray.at(id).get();
        newFuncTypeArgs.push_back(mlir::MemRefType::get(arr.shape(), numpy::dtype_to_mlir_type(arr.dtype, ctx)));
      }
      auto newFuncType = builder.getFunctionType(newFuncTypeArgs, std::nullopt);
      mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
      auto func = builder.create<mlir::func::FuncOp>(loc, "body", newFuncType, std::vector<mlir::NamedAttribute>{namedAttr});
      auto block = func.addEntryBlock();
      builder.setInsertionPointToStart(block);
      block->addArgument()

      // TODO (rohany): Need to also remove returns.

      // Now add each function body to the new function, and remap all of the arguments.
      for (auto& t : this->outstanding) {
        auto& taskModule = t.getMLIRBody();
        auto taskFunc = *taskModule->getOps<mlir::func::FuncOp>().begin();
        // TODO (rohany): What do I do if we have more than one block in the kernel?
        auto& taskBodyBlock = taskFunc.getBlocks().front();
        mlir::IRMapping irMapping;
        auto arrays = t.getArrays();
        for (size_t i = 0; i < arrays.size(); i++) {
          auto var = taskBodyBlock.getArgument(i);
          auto id = arrays[i].get().getID();
          auto newVarIdx = std::find(arrayIDs.begin(), arrayIDs.end(), id) - arrayIDs.begin();
          auto newVar = block->getArgument(newVarIdx);
          irMapping.map(var, newVar);
        }

        // Copy all of the instructions into the resulting block.
        for (auto& inst : taskBodyBlock) {
          if (!mlir::isa<mlir::func::ReturnOp>(inst)) {
            builder.clone(inst, irMapping);
          }
        }
      }

      builder.create<mlir::func::ReturnOp>(loc);

      ////////////////////////////////////////////////

      // This next stage of analysis circumvents current
      // limitations in MLIR around alias analysis. The
      // only form of alias analysis that MLIR performs
      // on memrefs is within a function. That means that
      // if two memrefs were alloc()'d in the same function,
      // then the analysis assumes that they are non-aliasing.
      // There's no way of specifying aliasing as function
      // arguments. To take advantage of this, we run optimizations
      // on an intermediate task representation where we convert
      // all array arguments to be memrefs allocated inside the
      // task body. This isn't enough however: when memrefs are
      // allocated inside a function, the analyses are free to
      // change the shape or remove intermediate allocations
      // from functions. Therefore, we must return memrefs that
      // cannot be modified from the function. Since we know
      // what arrays are "temporary", we can only return those
      // arrays that are non-temporary, and allow temporary
      // arrays to get eliminated.


      // Make a clone of the function to do some testing on.
      builder.setInsertionPointToEnd(module->getBody());
      std::vector<mlir::MemRefType> allArrTypes;
      std::vector<mlir::Type> persistentArrTypes;
      for (auto id : arrayIDs) {
        auto& arr = idToArray.at(id).get();
        auto memrefTy = mlir::MemRefType::get(arr.shape(), numpy::dtype_to_mlir_type(arr.dtype, ctx));
        allArrTypes.push_back(memrefTy);
        if (!arr.isTemporary()) {
          persistentArrTypes.push_back(memrefTy);
        }
      }
      auto modifiedFuncType = builder.getFunctionType(std::nullopt, persistentArrTypes);

      auto modifiedFunc = builder.create<mlir::func::FuncOp>(loc, "body2", modifiedFuncType, std::vector<mlir::NamedAttribute>{namedAttr});
      auto newBlock = modifiedFunc.addEntryBlock();
      builder.setInsertionPointToStart(newBlock);
      // Convert the arguments into memref allocations.
      std::vector<mlir::Value> arrayAllocs;
      std::vector<mlir::Value> persistentAllocs;
      for (size_t i = 0; i < allArrTypes.size(); i++) {
        auto alloc = builder.create<mlir::memref::AllocOp>(loc, allArrTypes[i]);
        arrayAllocs.push_back(alloc);
        if (!idToArray.at(arrayIDs[i]).get().isTemporary()) {
          persistentAllocs.push_back(alloc);
        }
      }
      mlir::IRMapping mapping;
      // Map the arguments to allocations.
      for (size_t i = 0; i < arrayIDs.size(); i++) {
        mapping.map(block->getArgument(i), arrayAllocs[i]);
      }

      for (auto& inst : *block) {
        if (mlir::isa<mlir::func::ReturnOp>(inst)) {
          builder.create<mlir::func::ReturnOp>(loc, persistentAllocs);
        } else {
          builder.clone(inst, mapping);
        }
      }

      func->remove();

      /////////////////////////////////////////////////

      auto mlirPassStart = std::chrono::high_resolution_clock::now();

      // Time to run optimizations on the MLIR!
      mlir::PassManager pm(module.get()->getName(), mlir::PassManager::Nesting::Implicit);
      // We want to run the loop fusion pass on functions nested inside the module,
      // so we express that by describing a nested pass manager.
      mlir::OpPassManager &funcsPM = pm.nest<mlir::func::FuncOp>();
//      funcsPM.addPass(mlir::createLinalgElementwiseOpFusionPass());
      funcsPM.addPass(mlir::createLoopFusionPass(0, 0, true, mlir::FusionMode::Greedy));
      funcsPM.addPass(mlir::createAffineScalarReplacementPass());
//      funcsPM.addPass(mlir::createConvertLinalgToAffineLoopsPass());
      // Run the registered passes, which modifies the operations in place.
      if (mlir::failed(pm.run(*module))) {
        assert(false);
      }

      auto mlirPassEnd = std::chrono::high_resolution_clock::now();

      std::cout << "MLIR Pass Exec Time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(mlirPassEnd - mlirPassStart)).count() << std::endl;

      // Now convert the hacked function back into the standard form, after the
      // first layer of MLIR optimizations have been completed.
      builder.setInsertionPointToEnd(module->getBody());
      {
        auto finalFuncType = builder.getFunctionType(persistentArrTypes, std::nullopt);
        auto finalFunc = builder.create<mlir::func::FuncOp>(loc, "kernel", finalFuncType, std::vector<mlir::NamedAttribute>{namedAttr});
        auto finalBlock = finalFunc.addEntryBlock();
        builder.setInsertionPointToStart(finalBlock);

        mapping.clear();
        size_t ctr = 0;
        for (auto inst : newBlock->getOps<mlir::memref::AllocOp>()) {
          mapping.map(inst, finalBlock->getArgument(ctr));
          ctr++;
          if (ctr == persistentArrTypes.size()) {
            break;
          }
        }

        // Finally transfer all of the operations over.
        for (auto& inst : *newBlock) {
          if (mlir::isa<mlir::memref::AllocOp>(inst)) {
            // Do nothing. To be more robust, we could copy over
            // alloc operations after the first |arrayID| allocs.
          } else if (mlir::isa<mlir::func::ReturnOp>(inst)) {
            builder.create<mlir::func::ReturnOp>(loc);
          } else {
            builder.clone(inst, mapping);
          }
        }
      }

      modifiedFunc->remove();

//      {
//        mlir::PassManager pm(module.get()->getName(), mlir::PassManager::Nesting::Implicit);
//        mlir::OpPassManager &funcsPM = pm.nest<mlir::func::FuncOp>();
//        funcsPM.addPass(mlir::createAffineParallelizePass());
//        funcsPM.addPass(mlir::createLowerAffinePass());
//        pm.addPass(mlir::createConvertSCFToOpenMPPass());
//        pm.addPass(mlir::createConvertOpenMPToLLVMPass());
//        if (mlir::failed(pm.run(*module))) {
//          assert(false);
//        }
//      }

      auto mlirBuildEnd = std::chrono::high_resolution_clock::now();

      std::cout << "MLIR build + pass Exec Time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(mlirBuildEnd - mlirBuildStart)).count() << std::endl;

      std::cout << "Final kernel being executed:" << std::endl;
      this->dumpMLIR(module.get());

      auto mlirToLLVMLowerStart = std::chrono::high_resolution_clock::now();

      // Now lower the code into LLVM.
      mlir::LLVMConversionTarget convTarget(*this->mlirContext);
      convTarget.addLegalOp<mlir::ModuleOp>();
      mlir::LLVMTypeConverter typeConverter(this->mlirContext.get());
      mlir::RewritePatternSet patterns(this->mlirContext.get());
      mlir::populateAffineToStdConversionPatterns(patterns);
      mlir::populateSCFToControlFlowConversionPatterns(patterns);
      mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
      mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
      mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
      mlir::populateMathToLLVMConversionPatterns(patterns);
//      mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);
      mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (mlir::failed(mlir::applyFullConversion(module.get(), convTarget, frozenPatterns))) {
        assert(false);
      }

      auto mlirToLLVMLowerEnd = std::chrono::high_resolution_clock::now();
      std::cout << "MLIR to llvm time : " << (std::chrono::duration_cast<std::chrono::milliseconds>(mlirToLLVMLowerEnd - mlirToLLVMLowerStart)).count() << std::endl;

      auto llvmJitStart = std::chrono::high_resolution_clock::now();
      // TODO (rohany): This is scoped to an individual operation. Is there a
      //  variant of the execution engine that can be re-used across operations?
      mlir::ExecutionEngineOptions engineOptions;
      engineOptions.transformer = this->llvmOptTransformer;
      // engineOptions.enableObjectDump = true;
      auto maybeEngine = mlir::ExecutionEngine::create(module.get(), engineOptions);
      assert(maybeEngine);
      auto& engine = maybeEngine.get();

      // Package up the deduplicated arrays into arguments.
      // Annoyingly, the execution engine voodoo on the inside
      // requires pointers to Memref objects, and then internally
      // takes a double pointer, meaning that Memref**'s must be
      // passed to invokePacked. So we have to play a dance here
      // of allocating memory for the double indirection before
      // handing it off to LLVM.
      std::vector<void*> argData;
      argData.reserve(persistentArrTypes.size());
      for (auto id : arrayIDs) {
        auto& arr = idToArray.at(id).get();
        if (arr.isTemporary()) {
          continue;
        }
        switch (arr.ndim) {
          case 1: {
            auto memref = arr.toMemref<double, 1>();
            auto argPtr = static_cast<std::add_pointer<decltype(memref)>::type>(malloc(sizeof(decltype(memref))));
            *argPtr = memref;
            argData.push_back(argPtr);
            break;
          }
          case 2: {
            auto memref = arr.toMemref<double, 2>();
            auto argPtr = static_cast<std::add_pointer<decltype(memref)>::type>(malloc(sizeof(decltype(memref))));
            *argPtr = memref;
            argData.push_back(argPtr);
            break;
          }
          case 3: {
            auto memref = arr.toMemref<double, 3>();
            auto argPtr = static_cast<std::add_pointer<decltype(memref)>::type>(malloc(sizeof(decltype(memref))));
            *argPtr = memref;
            argData.push_back(argPtr);
            break;
          }
          default:
            assert(false);
        }
      }
      std::vector<void*> args(argData.size());
      for (size_t i = 0; i < args.size(); i++) {
        args[i] = &argData[i];
      }

      auto start = std::chrono::high_resolution_clock::now();
      std::cout << "LLVM comp time : " << (std::chrono::duration_cast<std::chrono::milliseconds>(start - llvmJitStart)).count() << std::endl;

      auto result = engine->invokePacked("_mlir_ciface_kernel", args);
      if (result) {
        llvm::outs() << "JIT invocation failed. Error:\n" << result << "\n";
        assert(false);
      }
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

      std::cout << "Spent: " << duration.count() << std::endl;




      // Free all of the temporary memory.
      for (auto ptr : argData) {
        free(ptr);
      }

      this->outstanding.clear();
    }

    void dumpQueuedTasks() {
      for (auto& t : this->outstanding) {
        t.dumpMLIRBody();
      }
    }

    void enqueue(Task t) {
      this->outstanding.push_back(std::move(t));
      if (this->outstanding.size() == this->windowSize) {
        this->flush();
      }
    }

    void initializeMLIR() {
      // Register the necessary dialects and passes. This is a separate
      // step from _loading_ them, which will occur later.
      mlir::registerAllDialects(this->registry);
      mlir::registerAllPasses();
      mlir::registerBuiltinDialectTranslation(registry);
      mlir::registerLLVMDialectTranslation(registry);
      // Create the MLIRContext once all of the dialects and
      // passes have been registered.
      this->mlirContext = std::make_unique<mlir::MLIRContext>(this->registry, mlir::MLIRContext::Threading::DISABLED);
      // Now, we can load all of the dialects.
      this->mlirContext->loadAllAvailableDialects();
      this->parseConfig = std::make_unique<mlir::ParserConfig>(this->mlirContext.get(), true /* verify afterParse */, &this->fallbackResourceMap);

      // Initialize the LLVM JIT.
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();

      auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
      assert(tmBuilderOrError);
      auto tmOrError = tmBuilderOrError->createTargetMachine();
      assert(tmOrError);
      this->targetMachine = std::move(*tmOrError);
      // TODO (rohany): I also want to turn on associative floating point math, which i can't find the option for...
      // https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffast-math.
//      this->targetMachine->Options.UnsafeFPMath = true;
//      this->targetMachine->Options.NoInfsFPMath = true;
//      this->targetMachine->Options.NoNaNsFPMath = true;
//      this->targetMachine->Options.ApproxFuncFPMath = true;
//      this->targetMachine->Options.NoSignedZerosFPMath = true;

      this->llvmOptTransformer = mlir::makeOptimizingTransformer(3, 0, this->targetMachine.get());
    }

    std::unique_ptr<mlir::MLIRContext>& getMLIRContext() {
      return this->mlirContext;
    }

    std::unique_ptr<mlir::ParserConfig>& getMLIRParseConfig() {
      return this->parseConfig;
    }

    void dumpMLIR(mlir::Operation* op) {
      mlir::AsmState asmState(op, mlir::OpPrintingFlags(), nullptr /* locationMap */, &this->fallbackResourceMap);
      op->print(llvm::outs(), asmState);
      llvm::outs() << "\n";
    }

  private:
    std::vector<Task> outstanding;
    int64_t windowSize;

  private:
    // MLIR related state.
    mlir::DialectRegistry registry;
    std::unique_ptr<mlir::MLIRContext> mlirContext;
    mlir::FallbackAsmResourceMap fallbackResourceMap;
    std::unique_ptr<mlir::ParserConfig> parseConfig;
    std::unique_ptr<mlir::ExecutionEngine> mlirExecutionEngine;
    std::unique_ptr<llvm::TargetMachine> targetMachine;
    // LLVM related state.
    llvm::LLVMContext llvmContext;
    std::function<llvm::Error(llvm::Module*)> llvmOptTransformer;
  };
  static Runtime* runtime;

  // Additional definitions needed later, because I'm lazy and doing this
  // all in one file.
  void Task::dumpMLIRBody() {
    runtime->dumpMLIR(this->func->getOperation());
  }
}

namespace numpy {

  // API for NumPy functions.

  // TODO (rohany): Let's start with just add first.

  // Compute c = a + b. Writes into C.
  void add(Array& a, Array& b, Array& c);
  // Compute c = a * b. Writes into C.
  void mul(Array& a, Array& b, Array& c);
  // Fill an array with a value.
  void fill(Array& a, double v);

  // TODO (rohany): It seems like additional passes may be needed to support
  //  n-dimensional reductions, or fusion of affine.parallel regions.
  //  It all works correctly for iteration over 1-dimensional objects though.
  // Sum-Reduce a into the zero-dimensional (1-D of size 1) output array b.
//  void sumReduce(Array& a, Array& b) {
//    auto ctx = legate::runtime->getMLIRContext().get();
//    mlir::OpBuilder builder(ctx);
//    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
//    builder.setInsertionPointToEnd(module->getBody());
//    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "AAAAA"));
//
//    auto aType = mlir::MemRefType::get(a.shape(), numpy::dtype_to_mlir_type(a.dtype, ctx));
//    auto bType = mlir::MemRefType::get(b.shape(), numpy::dtype_to_mlir_type(b.dtype, ctx));
//    auto funcType = builder.getFunctionType({aType, bType}, std::nullopt);
//    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
//    auto func = builder.create<mlir::func::FuncOp>(loc, "body", funcType, std::vector<mlir::NamedAttribute>{namedAttr});
//    auto block = func.addEntryBlock();
//    auto aVar = block->getArgument(0);
//    auto bVar = block->getArgument(1);
//
//    builder.setInsertionPointToStart(block);
//
//    auto zero = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(0.0), builder.getF64Type());
//  }

  template <typename Op>
  mlir::OwningOpRef<mlir::ModuleOp> binopImpl(Array& a, Array& b, Array& c) {
    // TODO (rohany): There should be some easier way to build MLIR operations
    //  by just parsing, but then being able to generalize the necessary
    //  things like data types and dimensions.
    // The below code generates a function with IR similar to:
    // func.func @body(%a : memref<1000xf64>, %b : memref<1000xf64>, %c : memref<1000xf64>) attributes { llvm.emit_c_interface } {
    //   affine.for %i = 0 to 1000 {
    //     %0 = affine.load %a[%i] : memref<1000xf64>
    //     %1 = affine.load %b[%i] : memref<1000xf64>
    //     %2 = arith.OP %0, %1 : f64
    //     affine.store %2, %c[%i] : memref<1000xf64>
    //   }
    //   return
    // }
    auto ctx = legate::runtime->getMLIRContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "AAAAA"));
    mlir::OperationState state(loc, "what is this supposed to be?");

    auto aType = mlir::MemRefType::get(a.shape(), numpy::dtype_to_mlir_type(a.dtype, ctx));
    auto bType = mlir::MemRefType::get(b.shape(), numpy::dtype_to_mlir_type(b.dtype, ctx));
    auto cType = mlir::MemRefType::get(c.shape(), numpy::dtype_to_mlir_type(c.dtype, ctx));
    auto funcType = builder.getFunctionType({aType, bType, cType}, std::nullopt);
    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
    auto func = builder.create<mlir::func::FuncOp>(loc, "body", funcType, std::vector<mlir::NamedAttribute>{namedAttr});
    auto block = func.addEntryBlock();
    auto aVar = block->getArgument(0);
    auto bVar = block->getArgument(1);
    auto cVar = block->getArgument(2);

    builder.setInsertionPointToStart(block);
    mlir::buildAffineLoopNest(
        builder,
        loc,
        std::vector<int64_t>(a.ndim, 0),
        a.shape(),
        std::vector<int64_t>(a.ndim, 1),
        [&aVar, &bVar, &cVar](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange lvs) {
          auto aLoad = builder.create<mlir::AffineLoadOp>(loc, aVar, lvs);
          auto bLoad = builder.create<mlir::AffineLoadOp>(loc, bVar, lvs);
          auto add = builder.create<Op>(loc, aLoad.getType(), aLoad, bLoad);
          auto cStore = builder.create<mlir::AffineStoreOp>(loc, add, cVar, lvs);
        });
    builder.create<mlir::func::ReturnOp>(loc);

    return std::move(module);
  }

  template <typename Op>
  mlir::OwningOpRef<mlir::ModuleOp> binopLinalgImpl(Array& a, Array& b, Array& c) {
    auto ctx = legate::runtime->getMLIRContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "AAAAA"));
    mlir::OperationState state(loc, "what is this supposed to be?");

    auto aType = mlir::MemRefType::get(a.shape(), numpy::dtype_to_mlir_type(a.dtype, ctx));
    auto bType = mlir::MemRefType::get(b.shape(), numpy::dtype_to_mlir_type(b.dtype, ctx));
    auto cType = mlir::MemRefType::get(c.shape(), numpy::dtype_to_mlir_type(c.dtype, ctx));
    auto funcType = builder.getFunctionType({aType, bType, cType}, std::nullopt);
    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
    auto func = builder.create<mlir::func::FuncOp>(loc, "body", funcType, std::vector<mlir::NamedAttribute>{namedAttr});
    auto block = func.addEntryBlock();
    auto aVar = block->getArgument(0);
    auto bVar = block->getArgument(1);
    auto cVar = block->getArgument(2);

    builder.setInsertionPointToStart(block);

    std::vector<mlir::AffineExpr> affineExprs(a.ndim);
    mlir::bindDimsList<mlir::AffineExpr>(ctx, affineExprs);
    // This is a unary op.
    auto aMap = mlir::AffineMap::get(a.ndim, 0, affineExprs, ctx);
    auto bMap = mlir::AffineMap::get(b.ndim, 0, affineExprs, ctx);
    auto cMap = mlir::AffineMap::get(c.ndim, 0, affineExprs, ctx);
    std::vector<mlir::utils::IteratorType> iterators(a.ndim, mlir::utils::IteratorType::parallel);

    // TODO (rohany): Use the linalg dialect here.
    builder.create<mlir::linalg::GenericOp>(loc, mlir::ValueRange{aVar, bVar}, mlir::ValueRange{cVar}, mlir::ArrayRef<mlir::AffineMap>{aMap, bMap, cMap}, iterators,
                                            [](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange vals) {
                                              auto add = builder.create<Op>(loc, vals[0].getType(), vals[0], vals[1]);
                                              builder.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{add});
                                            });
    return std::move(module);
  }

  // Implementations of the NumPy API.
  void add(Array& a, Array& b, Array& c) {
    legate::Task task;
    task.addArray(a);
    task.addArray(b);
    task.addArray(c);
    assert(a.ndim == b.ndim && b.ndim == c.ndim);
    assert(a.dtype == b.dtype && b.dtype == c.dtype);
    auto module = binopImpl<mlir::arith::AddFOp>(a, b, c);
    task.setMLIRBody(std::move(module));
    legate::runtime->enqueue(std::move(task));
  }

  void addLinAlg(Array& a, Array& b, Array& c) {
    legate::Task task;
    task.addArray(a);
    task.addArray(b);
    task.addArray(c);
    assert(a.ndim == b.ndim && b.ndim == c.ndim);
    assert(a.dtype == b.dtype && b.dtype == c.dtype);
    auto module = binopLinalgImpl<mlir::arith::AddFOp>(a, b, c);
    task.setMLIRBody(std::move(module));
    legate::runtime->enqueue(std::move(task));
  }

  void mul(Array& a, Array& b, Array& c) {
    legate::Task task;
    task.addArray(a);
    task.addArray(b);
    task.addArray(c);
    assert(a.ndim == b.ndim && b.ndim == c.ndim);
    assert(a.dtype == b.dtype && b.dtype == c.dtype);
    auto module = binopImpl<mlir::arith::MulFOp>(a, b, c);
    task.setMLIRBody(std::move(module));
    legate::runtime->enqueue(std::move(task));
  }

  void mulLinAlg(Array& a, Array& b, Array& c) {
    legate::Task task;
    task.addArray(a);
    task.addArray(b);
    task.addArray(c);
    assert(a.ndim == b.ndim && b.ndim == c.ndim);
    assert(a.dtype == b.dtype && b.dtype == c.dtype);
    auto module = binopLinalgImpl<mlir::arith::MulFOp>(a, b, c);
    task.setMLIRBody(std::move(module));
    legate::runtime->enqueue(std::move(task));
  }

  // TODO (rohany): Do some type generalization later.
  void fill(Array& a, double val) {
    legate::Task task;
    task.addArray(a);
    auto ctx = legate::runtime->getMLIRContext().get();
    mlir::OpBuilder builder(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "AAAAA"));
    mlir::OperationState state(loc, "what is this supposed to be?");
    auto aType = mlir::MemRefType::get(a.shape(), numpy::dtype_to_mlir_type(a.dtype, ctx));
    auto funcType = builder.getFunctionType({aType}, std::nullopt);
    mlir::NamedAttribute namedAttr(mlir::StringAttr::get(ctx, "llvm.emit_c_interface"), mlir::UnitAttr::get(ctx));
    auto func = builder.create<mlir::func::FuncOp>(loc, "body", funcType, std::vector<mlir::NamedAttribute>{namedAttr});
    auto block = func.addEntryBlock();
    auto aVar = block->getArgument(0);
    builder.setInsertionPointToStart(block);
    mlir::buildAffineLoopNest(
        builder,
        loc,
        std::vector<int64_t>(a.ndim, 0),
        a.shape(),
        std::vector<int64_t>(a.ndim, 1),
        [&aVar, &val](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange lvs) {
          auto constant = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(val), builder.getF64Type());
          auto store = builder.create<mlir::AffineStoreOp>(loc, constant, aVar, lvs);
          mlir::arith::ConstantIndexOp
        });
    builder.create<mlir::func::ReturnOp>(loc);
    task.setMLIRBody(std::move(module));
    legate::runtime->enqueue(std::move(task));
  }
}


int main(int argc, char** argv) {
  int windowSize = 1;
  if (argc == 2) {
    windowSize = atoi(argv[1]);
  }

  legate::runtime = new legate::Runtime(windowSize);
  legate::runtime->initializeMLIR();

  // Starter program -- declare some arrays!
  constexpr int64_t n = 20000;
  std::vector<int64_t> dims = {n, n};
  numpy::Array a(numpy::Double, dims),
               b(numpy::Double, dims),
               c(numpy::Double, dims),
               d(numpy::Double, dims),
               temp1(numpy::Double, dims),
               temp2(numpy::Double, dims),
               temp3(numpy::Double, dims),
               temp4(numpy::Double, dims);

  numpy::fill(a, 1.0);
  legate::runtime->flush();

  numpy::add(a, a, temp1);
  numpy::add(a, temp1, temp2);
  numpy::add(a, temp2, temp3);
  numpy::add(a, temp3, temp4);
  numpy::add(a, temp4, b);
  numpy::mul(a, b, b);
  numpy::add(a, b, b);
  numpy::add(a, b, b);
  numpy::add(a, b, b);
  numpy::mul(a, b, b);
  numpy::add(a, b, b);
  numpy::add(a, b, b);
  numpy::add(a, b, b);
  numpy::mul(a, b, b);
  numpy::add(a, b, b);
  numpy::mul(a, b, b);
  numpy::mul(a, b, b);
  numpy::add(a, b, b);
  numpy::add(a, b, b);
  numpy::mul(a, b, b);
  numpy::mul(a, b, b);
  numpy::add(a, b, b);
  numpy::add(a, b, b);
  numpy::add(a, b, b);

  if (windowSize > 1) {
    temp1.setTemporary(true);
    temp2.setTemporary(true);
    temp3.setTemporary(true);
    temp4.setTemporary(true);
  }

//  numpy::addLinAlg(a, b, temp);
//  numpy::mulLinAlg(temp, c, d);
//  numpy::add(a, b, temp);
//  legate::runtime.flush();
//  numpy::add(temp, c, temp);
//  numpy::mul(temp, b, temp);

  legate::runtime->flush();

//  numpy::fill(a, 1.0);
//  numpy::fill(b, 2.0);
//  legate::runtime.flush();
//
//  numpy::add(a, b, temp);
//  numpy::add(temp, c, temp);
//  numpy::add(temp, d, temp);
//  numpy::mul(temp, d, temp);

  // Do some "numpy" operations now.
//  numpy::fill(a, 1.0);
//  numpy::fill(b, 2.0);
//  numpy::add(a, b, c);
//  numpy::add(b, c, d);
//  numpy::mul(c, d, d);
//
//  legate::runtime.flush();
//
//  // Check that the results are still correct.
//  auto cref = c.toMemref<double, 2>();
//  auto dref = d.toMemref<double, 2>();
//  for (int64_t i = 0; i < n; i++) {
//    for (int64_t j = 0; j < n; j++) {
//      assert(cref[i][j] == 3.0);
//      assert(dref[i][j] == 15.0);
//    }
//  }

  return 0;
}
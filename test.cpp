#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
// Have to include this to avoid getting a forward declaration error.
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  // Register a command line option for us.
  static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));
  ParseCommandLineOptions(argc, argv, "testing!");

  // Read in the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    std::cout << errorMessage << std::endl;
    return 1;
  }

  // Construct an MLIR Context under which everything will happen.
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();

  // Set up a parsing configuration.
  mlir::FallbackAsmResourceMap fallbackResourceMap;
  mlir::ParserConfig parseConfig(&context, true /* verifyAfterParse */, &fallbackResourceMap);

  // Load the file buffer into a "SourceManager".
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  // Actually parse something.
  mlir::OwningOpRef<mlir::Operation*> op = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parseConfig);

  std::cout << "INITIAL FUNCTION: " << std::endl;
  {
    mlir::AsmState asmState(op.get(), mlir::OpPrintingFlags(), nullptr /* locationMap */, &fallbackResourceMap);
    op.get()->print(llvm::outs(), asmState);
    std::cout << std::endl;
  }

  // Set up a pass manager.
  mlir::PassManager pm(op.get()->getName(), mlir::PassManager::Nesting::Implicit);
  // We want to run the loop fusion pass on functions nested inside the module,
  // so we express that by describing a nested pass manager.
  mlir::OpPassManager &funcsPM = pm.nest<mlir::func::FuncOp>();
  funcsPM.addPass(mlir::createLoopFusionPass(0, 0, true, mlir::FusionMode::Greedy));
  funcsPM.addPass(mlir::createAffineScalarReplacementPass());

  // Run the registered passes, which modifies the operations in place.
  if (mlir::failed(pm.run(*op))) {
    std::cout << "Pass manager failed." << std::endl;
  }

  std::cout << "OPTIMIZED FUNCTION: " << std::endl;
  {
    mlir::AsmState asmState(op.get(), mlir::OpPrintingFlags(), nullptr /* locationMap */, &fallbackResourceMap);
    op.get()->print(llvm::outs(), asmState);
    std::cout << std::endl;
  }

  // Let's try to lower to LLVM code now. This is adapted from Toy chapter 7.
  mlir::LLVMConversionTarget convTarget(context);
  convTarget.addLegalOp<mlir::ModuleOp>();
  mlir::LLVMTypeConverter typeConverter(&context);

  mlir::RewritePatternSet patterns(&context);
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // Don't have any custom operations that need additional lowering out of MLIR.

  if (mlir::failed(mlir::applyFullConversion(op.get(), convTarget, std::move(patterns)))) {
    return 1;
  }

  std::cout << "Successfully converted to LLVM dialect!" << std::endl;

  // Now let's try and dump out the LLVM.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(op.get(), llvmContext);

  std::cout << "Result of conversion to LLVM:" << std::endl;

  llvm::outs() << (*llvmModule) << "\n";

  // Let's also run an optimization pass over the LLVM IR.

  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::outs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }

  std::cout << "Result of optimizing LLVM:" << std::endl;

  llvm::outs() << (*llvmModule) << "\n";

  std::cout << "Attempting to JIT LLVM to native code and execute it." << std::endl;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::ExecutionEngineOptions engineOptions;
  auto maybeEngine = mlir::ExecutionEngine::create(op.get(), engineOptions);
  assert(maybeEngine);
  auto& engine = maybeEngine.get();

  // Allocate a buffer that we'll pass to the MLIR generated code.
  constexpr int n = 10;
  std::vector<int32_t> data(10);
  for (int i = 0; i < n; i++) {
    data[i] = 0;
  }
  // Create a memref that references this data.
  StridedMemRefType<int32_t, 1> dataMemRef{};
  dataMemRef.basePtr = data.data();
  dataMemRef.data = data.data();
  dataMemRef.offset = 0;
  dataMemRef.sizes[0] = n;
  dataMemRef.strides[0] = 1;

  auto result = engine->invoke("weija", &dataMemRef);
  if (result) {
    llvm::outs() << "JIT invocation failed. Error:\n" << result << "\n";
    return -1;
  }

  std::cout << "{";
  for (int i = 0; i < n; i++) {
    if (i > 0) std::cout << ", ";
    std::cout << data[i];
  }
  std::cout << "}" << std::endl;

  return 0;
}

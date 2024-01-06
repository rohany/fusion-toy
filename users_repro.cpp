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


class TestingPass : public mlir::PassWrapper<TestingPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
  TestingPass() {}
  void runOnOperation() final {
    auto& ctx = this->getContext();
    mlir::func::FuncOp op = this->getOperation();
    auto& entryBlock = op.getBlocks().front();
    auto numArgs = entryBlock.getNumArguments();

    for (int32_t i = 0; i < numArgs; i++) {
      auto arg = entryBlock.getArgument(i);
      std::cout << "Users of argument " << i << std::endl;
      for (auto user : arg.getUsers()) {
        user->dump();
      }
    }
  }
};

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

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

  // Set up a pass manager.
  mlir::PassManager pm(op.get()->getName(), mlir::PassManager::Nesting::Implicit);
  // We want to run the loop fusion pass on functions nested inside the module,
  // so we express that by describing a nested pass manager.
  mlir::OpPassManager &funcsPM = pm.nest<mlir::func::FuncOp>();
  funcsPM.addPass(std::make_unique<TestingPass>());

  // Run the registered passes, which modifies the operations in place.
  if (mlir::failed(pm.run(*op))) {
    std::cout << "Pass manager failed." << std::endl;
  }

  return 0;
}
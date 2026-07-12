from dataclasses import dataclass
import subprocess
import sys
import tempfile

@dataclass
class ExecutionResult:
    success: bool
    stdout: str | None
    stderr: str | None
    timed_out: bool

def run_code(code: str, timeout: int) -> ExecutionResult:

    with tempfile.TemporaryDirectory() as scratch_dir:

        try:
            result = subprocess.run([sys.executable, "-c", code], 
                                    timeout=timeout, 
                                    text=True, 
                                    capture_output=True,
                                    cwd=scratch_dir, # subprocess runs with this as its working directory
                                    env={}, # subprocess gets s stripped/empty environment
                                    )
            
        except subprocess.TimeoutExpired as e:
            return ExecutionResult(success=False, 
                                timed_out=True, 
                                stderr=str(e), 
                                stdout=None)
        else:
            if result.stderr:
                return ExecutionResult(success=False,
                                    timed_out=False,
                                    stderr=result.stderr,
                                    stdout=None)
            else:
                return ExecutionResult(success=True,
                                    timed_out=False,
                                    stderr=None,
                                    stdout=result.stdout)
        


    




if __name__ == "__main__":

    result = run_code('print(2+2)', timeout=5)
    print(result)



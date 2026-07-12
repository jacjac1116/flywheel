from dataclasses import dataclass
import subprocess
import sys
import tempfile
import shutil
from analyst_agent.core import get_configs

@dataclass
class ExecutionResult:
    success: bool
    stdout: str 
    stderr: str 
    timed_out: bool
    code: str

def run_code(code: str, timeout: int) -> ExecutionResult:

    with tempfile.TemporaryDirectory() as scratch_dir:

        configs = get_configs()
        carbon_data = configs['carbon_data']
        # Copy the content of path into scratch_dir
        shutil.copy2(carbon_data, scratch_dir)

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
                                stdout="",
                                code=code)
        else:
            if result.stderr:
                return ExecutionResult(success=False,
                                    timed_out=False,
                                    stderr=result.stderr,
                                    stdout=result.stdout,
                                    code=code)
            else:
                return ExecutionResult(success=True,
                                    timed_out=False,
                                    stderr=result.stderr,
                                    stdout=result.stdout,
                                    code=code)
        


    




if __name__ == "__main__":

    result = run_code('print(2+2)', timeout=5)
    print(result)



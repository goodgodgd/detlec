import os
import shutil


class MyExceptionToCatch(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class PathManager:
    def __init__(self, path, closer_func=None):
        self.path = path
        self.safe_exit = False
        self.closer = closer_func

    def __enter__(self):
        os.makedirs(self.path, exist_ok=True)
        return self

    def reopen(self, path, closer_func=None):
        self.path = path
        self.safe_exit = False
        self.closer = closer_func
        os.makedirs(path, exist_ok=True)
        return self

    def set_ok(self):
        self.safe_exit = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.closer:
            self.closer()

        if self.safe_exit is False:
            print("[PathManager] the process is NOT ended properly, remove the working path")
            if self.path is not None and os.path.isdir(self.path):
                print("    remove:", self.path)
                shutil.rmtree(self.path)
            # to ensure the process stop here
            assert False


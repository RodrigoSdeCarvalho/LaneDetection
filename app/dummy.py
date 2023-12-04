from essentials.add_module import set_working_directory
set_working_directory()

from utils.path import Path
from utils.logger import Logger


# Dummy test for set_working_directory and how it allows relative imports and singletons/class variables to work.

def dummy_main():
    dummy_path = Path()
    Logger.info('This is a dummy path')
    print(dummy_path.root)
    Logger.error('This is a dummy path')

    print(dummy_path.assets)
    Logger.trace('This is a dummy path')
    print(dummy_path.data)
    Logger.warning('This is a dummy path')
    print(dummy_path.models)
    print(dummy_path('assets', 'dummy.txt'))
    Logger().log("HGHGHGH", Logger.Type.INFO)


if __name__ == '__main__':
    dummy_main()

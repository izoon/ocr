import re

def safe_import(module_name):
    try:
        if module_name == 'opencv-python':
            module = __import__('cv2')
        elif module_name == 'Pillow':
            module = __import__('PIL')
        else:
            module = __import__(module_name)
        print(f'{module_name}: {module.__version__}')
    except ImportError:
        print(f'{module_name}: NOT INSTALLED')
    except AttributeError:
        print(f'{module_name}: NO VERSION')

# Read requirements.txt and extract package names
with open('requirements.txt', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#') and not line.startswith('--'):
            package = re.split('>=|<=|==', line.strip())[0]
            safe_import(package)


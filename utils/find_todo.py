from pathlib import Path 
import os

if __name__ == '__main__':
    all_files = list(Path('../src').glob('**/*.*'))
    all_files += list(Path('../include').glob('**/*.*'))
    for path in all_files:
        with open(str(path)) as f:
            for num, line in enumerate(f, 1):
                if 'TODO' in line:
                    name = os.path.basename(f.name)
                    idx = line.index("TODO") + 4
                    print("Found TODO (" + line[idx:].strip() + ") in " + name + ":" + str(num))

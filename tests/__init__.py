from pathlib import Path
import sys

directory = Path(__file__).parent.parent

sys.path.append((directory/ "vidmd").absolute())
# print((directory/ "vidmd").absolute())
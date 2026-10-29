import argparse
from core import *

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['plan','check']); a=p.parse_args()
    if a.command=='plan':
        prepare(Path('/study'),Path('/prior'),Path('/audit'),Path('/acdc'),Path('/train'),Path('/validation'),Path('/weights'),read('/prior/local-runtime-images.json'))
    else: checked(Path('/study')); print('SOURCE VERIFIED',flush=True)

import argparse
from common import *

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['plan','train','infer','record','report','diagnose','diagnosis-report'])
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--arm',choices=['pretrained','M0','W1','D1','G1'],default='G1')
    p.add_argument('--branch',choices=['points','g1'],default='points')
    p.add_argument('--folder')
    a=p.parse_args(); root=Path('/study'); previous=Path('/previous')
    if a.command=='plan': prepare(root,previous)
    elif a.command=='train':
        from train_g1 import run
        run(root,previous,Path('/weights'),Path('/audit'),Path('/acdc'),Path('/data'),a.seed)
    elif a.command in ('diagnose','diagnosis-report'):
        import diagnose
        if a.command=='diagnose': diagnose.run(root,previous,Path('/data'),Path('/weights'),a.seed)
        else: diagnose.report(root,previous)
    else:
        import evaluate
        if a.command=='infer': evaluate.infer(root,previous,Path('/data'),Path('/weights'),a.arm,a.seed)
        elif a.command=='record': evaluate.record(root,previous,a.folder)
        else: evaluate.report(root,previous,a.branch)

"""Diagnostic timing scopes for the SHA-verified trainer, without editing it.

AST instrumentation adds synchronization/timers and an early diagnostic stop.
It does NOT change num_steps used by the optimizer/scheduler, losses or RNG.
Synchronized timings are intrusive diagnostics, not production throughput.
"""
import ast
from contextlib import contextmanager
import time
import torch


class PhaseProbe:
    def __init__(self,limit):
        self.limit=limit
        self.rows=[]; self.current=None; self.stack=[]; self.pending_data=0.
        self.outside_steps={}
        self.timing_errors=[]
        self.on_row=None

    def finish_sync(self):
        # Preserve the original training exception if a failed CUDA context
        # also makes timing synchronization fail; never claim completeness.
        try: self.sync()
        except Exception as error: self.timing_errors.append(str(error))

    def sync(self):
        if torch.cuda.is_available(): torch.cuda.synchronize()

    def memory(self):
        if not torch.cuda.is_available(): return {'cuda':False}
        free,total=torch.cuda.mem_get_info()
        return {'cuda':True,'allocated_mib':torch.cuda.memory_allocated()/1048576,
                'reserved_mib':torch.cuda.memory_reserved()/1048576,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1048576,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1048576,
                'driver_free_mib':free/1048576,'driver_total_mib':total/1048576}

    @contextmanager
    def stage(self,name):
        self.sync()
        frame={'start':time.perf_counter(),'child':0.}
        self.stack.append(frame)
        try: yield
        finally:
            self.finish_sync()
            elapsed=time.perf_counter()-frame['start']
            self.stack.pop()
            if self.stack: self.stack[-1]['child']+=elapsed
            bucket=self.current['phases'] if self.current is not None else self.outside_steps
            item=bucket.setdefault(name,{'calls':0,'inclusive_seconds':0.,'exclusive_seconds':0.})
            item['calls']+=1; item['inclusive_seconds']+=elapsed
            item['exclusive_seconds']+=max(0.,elapsed-frame['child'])

    @contextmanager
    def step(self,index):
        if self.current is not None: raise ValueError('Nested diagnostic steps')
        self.sync()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        row={'step':int(index),'phases':{},'data_load_seconds':self.pending_data,
             'optimizer_updates':0,'complete':False,'memory_before':self.memory()}
        self.pending_data=0.
        self.current=row; start=time.perf_counter()
        try:
            yield
            row['complete']=True
        finally:
            self.finish_sync()
            row['wall_seconds_including_data']=time.perf_counter()-start+row['data_load_seconds']
            try: row['memory_after']=self.memory()
            except Exception as error:
                row['memory_after']={'error':str(error)}
                self.timing_errors.append(str(error))
            self.rows.append(row); self.current=None
            if self.on_row: self.on_row()

    def bind(self,output):
        if self.current is not None:
            info=output['teacher_sampling']
            self.current['case_names']=info['case_names']
            self.current['primary_index']=info['primary_index']
            self.current['queries_sha256']=info['queries_sha256']

    def mark_update(self,index):
        if self.current is not None:
            self.current['optimizer_updates']+=1
            self.current['next_step']=int(index)

    @contextmanager
    def installed(self):
        from cotracker.models.core.cotracker.feature_sidecar import FeatureSidecar
        from torch.utils.data.dataloader import _BaseDataLoaderIter
        residual,next_batch=FeatureSidecar.residual,_BaseDataLoaderIter.__next__
        probe=self
        def timed_residual(branch,hidden,rich):
            if probe.current is not None:
                probe.current['sidecar_frames']=hidden.shape[2]
                probe.current['sidecar_points']=hidden.shape[1]
            with probe.stage('sidecar_forward'): return residual(branch,hidden,rich)
        def timed_next(iterator):
            start=time.perf_counter()
            try: return next_batch(iterator)
            finally: probe.pending_data+=time.perf_counter()-start
        FeatureSidecar.residual=timed_residual
        _BaseDataLoaderIter.__next__=timed_next
        try: yield
        finally:
            FeatureSidecar.residual=residual
            _BaseDataLoaderIter.__next__=next_batch


def call_name(node):
    if isinstance(node,ast.Call): return ast.unparse(node.func)
    return ''


class InstrumentTrainer(ast.NodeTransformer):
    def __init__(self): self.function=''; self.counts={}

    def wrap(self,node,name):
        self.counts[name]=self.counts.get(name,0)+1
        context=ast.Call(ast.Attribute(ast.Name('__probe__',ast.Load()),'stage',ast.Load()),[ast.Constant(name)],[])
        return ast.copy_location(ast.With([ast.withitem(context,None)],[node]),node)

    def visit_FunctionDef(self,node):
        previous=self.function; self.function=node.name
        node=self.generic_visit(node); self.function=previous
        if node.name=='save_training_checkpoint':
            block=ast.Pass(); block.lineno=node.lineno; block.col_offset=node.col_offset
            scope=self.wrap(block,'checkpoint_save'); scope.body=node.body; node.body=[scope]
        return node

    def visit_If(self,node):
        node=self.generic_visit(node)
        if self.function=='_forward_batch_single_teacher' and ast.unparse(node.test)=="'cotracker_three' in teacher_model_type":
            return self.wrap(node,'teacher_forward')
        if self.function=='run' and ast.unparse(node.test)=='self.global_rank == 0':
            names={n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
            constants={n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
            if 'FINISHED TRAINING' in constants: return self.wrap(node,'finalize_io')
            if 'video_array' in names: return self.wrap(node,'audit_and_checkpoint_io')
            if 'logger' in names and 'output' in names: return self.wrap(node,'logging_io')
        return node

    def visit_Compare(self,node):
        if (self.function=='run' and isinstance(node.left,ast.Name) and node.left.id=='total_steps'
            and len(node.ops)==1 and isinstance(node.ops[0],ast.GtE)
            and ast.unparse(node.comparators[0])=='args.num_steps'):
            node.comparators[0]=ast.Attribute(ast.Name('__probe__',ast.Load()),'limit',ast.Load())
            self.counts['early_stop']=self.counts.get('early_stop',0)+1
        return self.generic_visit(node)

    def visit_For(self,node):
        node=self.generic_visit(node)
        if self.function=='run' and ast.unparse(node.target)=='(i_batch, batch)':
            context=ast.Call(ast.Attribute(ast.Name('__probe__',ast.Load()),'step',ast.Load()),[ast.Name('total_steps',ast.Load())],[])
            node.body=[ast.With([ast.withitem(context,None)],node.body)]
            self.counts['training_step']=self.counts.get('training_step',0)+1
        elif self.function=='run' and ast.unparse(node.target)=='(k, v)' and ast.unparse(node.iter)=='output.items()':
            return self.wrap(node,'loss_or_logging_loop')
        return node

    def visit_Assign(self,node):
        node=self.generic_visit(node)
        name=call_name(node.value)
        if self.function=='_forward_batch_single_teacher' and name=='model': return self.wrap(node,'student_forward')
        if self.function=='run' and name=='forward_batch':
            bind=ast.Expr(ast.Call(ast.Attribute(ast.Name('__probe__',ast.Load()),'bind',ast.Load()),[ast.Name('output',ast.Load())],[]))
            return [self.wrap(node,'forward_batch'),ast.copy_location(bind,node)]
        return node

    def visit_AugAssign(self,node):
        if self.function=='run' and isinstance(node.target,ast.Name) and node.target.id=='total_steps':
            mark=ast.Expr(ast.Call(ast.Attribute(ast.Name('__probe__',ast.Load()),'mark_update',ast.Load()),[ast.Name('total_steps',ast.Load())],[]))
            return [node,ast.copy_location(mark,node)]
        return self.generic_visit(node)

    def visit_Expr(self,node):
        node=self.generic_visit(node)
        if self.function!='run': return node
        names={'dataclass_to_cuda_':'host_to_device','self.backward':'backward_total',
               'self.barrier':'barrier','optimizer.zero_grad':'zero_grad',
               'scaler.unscale_':'optimizer_update','torch.nn.utils.clip_grad_norm_':'optimizer_update',
               'scaler.step':'optimizer_update','scheduler.step':'optimizer_update','scaler.update':'optimizer_update'}
        name=call_name(node.value)
        return self.wrap(node,names[name]) if name in names else node


def instrument(source,filename):
    transform=InstrumentTrainer()
    tree=transform.visit(ast.parse(source,filename=filename))
    required={'teacher_forward':1,'student_forward':2,'training_step':1,'early_stop':2,
              'host_to_device':1,'forward_batch':1,'backward_total':1,'checkpoint_save':1}
    for name,count in required.items():
        if transform.counts.get(name)!=count: raise ValueError('Unrecognized trainer timing anchor: '+name)
    return compile(ast.fix_missing_locations(tree),filename,'exec'),transform.counts

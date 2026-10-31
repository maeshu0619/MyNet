"""Run real private training with den6 profiles/cache/worker access forbidden.

The native SparsePCGC GT observations and the autonomous spatial generator
remain available. This distinguishes codec observations from editor labels.
"""
import contextlib
import json
from pathlib import Path
import runpy
import sys
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from models.utils.pointcloud import ana_den6_online as proposals
    from models.modules import heuristic_guidance as guidance
    out=Path(sys.argv[sys.argv.index('--out')+1])
    def forbidden(*args,**kwargs):
        raise AssertionError('Autonomous training attempted to read den6 analysis/profile/cache')
    class ForbiddenProfile(dict):
        def get(self,*args,**kwargs):return forbidden(*args,**kwargs)
        def __getitem__(self,key):return forbidden(key)
    observed={}
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(proposals,'_FAST_PROFILES',ForbiddenProfile()))
        for name in ('_load_exact_single_plan_teacher','_load_or_build_payload'):
            observed[name]=stack.enter_context(mock.patch.object(proposals,name,side_effect=forbidden))
        observed['resolve_profile']=stack.enter_context(mock.patch.object(guidance,'resolve_profile',side_effect=forbidden))
        runpy.run_path(str(ROOT/'experiments/autonomous_speed_probe.py'),run_name='__main__')
    with (out/'den6_independence.json').open('x') as stream:
        json.dump(dict(completed=True,forbidden_calls={k:v.call_count for k,v in observed.items()},
                       native_codec_observation_allowed=True),stream,indent=2)


if __name__=='__main__':main()

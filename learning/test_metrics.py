"""Test statistical reporting rather than drawing implementation details."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from summarize_benchmarks import summarize
from build_dashboard import build


def report(label):
    return dict(label=label,policy='checkpoint' if label=='trained' else label,
        completed=True,protocol=dict(seeds=[11,22,33],decisions=6,sim_budget=180),
        episodes=[dict(seed=seed,steps=[],return_sum=value,area_gain_m2=value,
            sim_seconds=60,navigation_success_rate=.5,area_per_sim_minute=value,
            error=None,stop_reason='step_limit') for seed,value in [(11,1),(22,2),(33,3)]])


class MetricsTests(unittest.TestCase):
    def test_failures_remain_in_mean_and_pairing_requires_same_protocol(self):
        trained=report('trained')
        base=report('random')
        trained['episodes'][0].update(return_sum=0,error='sensor failure',stop_reason='runtime_error')
        result=summarize([trained,base])
        group=result['groups'][0]
        self.assertAlmostEqual(group['metrics']['return_sum']['mean'],5/3)
        self.assertEqual(group['errors'],1)
        self.assertEqual(group['interrupted'],1)
        self.assertEqual(result['paired_differences'][0]['return_difference']['n'],3)
        self.assertFalse(result['convergence_claim'])
        base['protocol']['decisions']=10
        self.assertEqual(summarize([trained,base])['paired_differences'],[])

    def test_missing_diagnostics_not_invented_and_html_payload_is_escaped(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            training=root/'training.json'
            training.write_text(json.dumps(dict(method='</script><script>bad()</script>',
                metrics=[dict(update=1,loss=2.)])),encoding='utf-8')
            out=root/'monitor'
            build(root,training,out)
            data=json.loads((out/'monitor-data.json').read_text(encoding='utf-8'))
            self.assertNotIn('q_mean',data['training']['metrics'][0])
            text=(out/'index.html').read_text(encoding='utf-8')
            self.assertNotIn('</script><script>bad()',text)
            self.assertNotIn('__MONITOR_DATA__',text)


if __name__=='__main__':
    unittest.main()

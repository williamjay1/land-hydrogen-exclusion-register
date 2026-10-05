"""Cross-platform entry-point and scientific mechanism regression tests."""
from pathlib import Path
import importlib.util
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_reproduce', ROOT/'scripts/reproduce.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
    def test_legacy_paths_and_escape(self):
        self.assertEqual(entry.relocate(r'D:\MLWork\Europe_green_hydrogen_siting\results\x.csv'), ROOT/'results/x.csv')
        self.assertEqual(entry.relocate('results/x.csv'), ROOT/'results/x.csv')
        with self.assertRaises(ValueError): entry.relocate('../escape.csv')
        with self.assertRaises(ValueError): entry.relocate(r'E:\unapproved\x.csv')

    def test_mechanisms(self):
        result = entry.register_module().mechanism_self_test()
        self.assertEqual(result['multihop_reopen'], 'PASS')
        self.assertEqual(result['cycle_rejection'], 'PASS')
        self.assertEqual(result['tiny_context_nontransitivity_test']['strict_retained'], [True,True,True])

    def test_primary_withdrawal(self):
        m = entry.register_module()
        _, manifest, records, members = m.load_built(ROOT)
        rr, mm = [r for r in records if m.in_main(r)], [r for r in members if m.in_main(r)]
        revoked = m.transition(rr, mm, {'DEA29':'revoked'})
        unknown = m.transition(rr, mm, {'DEA29':'unknown'})
        affected = lambda r: sorted(x['removed'] for x in r['exclusions'] if x['effective_status'] != 'EXCLUDED')
        self.assertEqual(len(affected(revoked)),13)
        self.assertEqual(affected(revoked), affected(unknown))
        self.assertEqual(manifest['main_scope']['initial_excluded'],38)

    def test_invalidated_scope(self):
        m = entry.register_module()
        _, _, records, members = m.load_built(ROOT)
        rr, mm = [r for r in records if m.in_main(r)], [r for r in members if m.in_main(r)]
        changed = m.transition(rr, mm, {}, ['preference'])
        self.assertEqual(sum(r['effective_status'] != 'EXCLUDED' for r in changed['exclusions']),38)


if __name__ == '__main__': unittest.main()

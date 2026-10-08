"""CPU-only production workunit inspector tests; no live BOINC access."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import boinc_cpu_workunit as inspector
from validator import load_bundle


CATALOG = ('1/0,26/0,15/0,37/1,67/0,104/0,12/1,231/0,255/1\n'
           '1/0,26/0,15/0,37/1,67/0,104/0,34/1,231/0,255/1\n'
           '1/0,26/0,15/0,37/1,67/0,104/0,36/1,197/2,255/1\n')
HINTS = '249 2 2 2\n181 2 13 1\n139 7 8 0\n255 13 2 1\n208 13 13 1\n'
COMMAND = ('--force-interleave --shard-index 1 --shard-count 4 '
           '--force-jitter-start 10 --force-jitters 14 --node-cap 2000000000 '
           '--seconds 7200 --min-save 463 --root-allow exploration_allow.txt '
           '--seed 6152026 --epoch cloudflare-boinc-explore')


def job_xml(command=COMMAND):
    return ('<job_desc><task><application>bw_runner</application>'
            '<command_line>' + command + '</command_line></task></job_desc>')


class CpuWorkunitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cpu-workunit-test-')
        self.root = Path(self.temp.name)
        self.slot = self.root / 'slot'
        self.project = self.root / 'project'
        self.slot.mkdir(); self.project.mkdir()
        self.write('job.xml', job_xml())
        self.write('256pieces.txt', ''.join(' '.join(map(str, row)) + '\n' for row in self.bundle.pieces_udlr))
        self.write('campaign_catalog.txt', CATALOG)
        self.write('campaign_hints.txt', HINTS)
        self.write('exploration_allow.txt', '0\n1\n2\n')
        self.manifest = {'campaign_revision': 65, 'frame_k': 2, 'root_set_id': 'test-frame2-r65',
                         'expected': {key: hashlib.sha256((self.slot / name).read_bytes()).hexdigest()
                                      for key, name in [('hints_sha256', 'campaign_hints.txt'),
                                                        ('root_allow_sha256', 'exploration_allow.txt'),
                                                        ('root_catalog_sha256', 'campaign_catalog.txt')]}}
        self.write('campaign_manifest.json', json.dumps(self.manifest))

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text):
        (self.slot / name).write_bytes(text.encode('utf-8'))

    def inspect(self, **kwargs):
        return inspector.inspect_workunit(self.slot, bundle=self.bundle, **kwargs)

    def test_reads_existing_ids_without_writes(self):
        before = {p.name: p.read_bytes() for p in self.slot.iterdir()}
        unit = self.inspect()
        tickets = list(unit.iter_tickets())
        self.assertEqual([(t.globalticket, t.jidx, t.cidx) for t in tickets], [(31, 10, 1), (35, 11, 2), (39, 13, 0)])
        self.assertEqual(unit.ticket_count, 3)
        self.assertEqual(unit.frame_rotation_cw, 2)
        self.assertEqual(unit.job.epoch, 'cloudflare-boinc-explore')
        self.assertEqual(tickets[0].prefix_states, unit.catalog[1])
        self.assertEqual(set(dict(unit.manifest_hash_checks).values()), {'raw-match'})
        report = unit.summary(3)
        self.assertTrue(report['read_only']); self.assertTrue(report['prefix_geometry_verified'])
        self.assertFalse(report['cpu_cfg_hash_recomputed'])
        self.assertFalse(report['production_dfs_compatibility_verified'])
        self.assertTrue(report['tickets_sample_is_complete'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.slot.iterdir()})

    def test_launcher_inspection_route_is_read_only(self):
        import launcher
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            code = launcher.main(['inspect-cpu', '--slot', str(self.slot), '--tickets', '1'])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report['read_only'])
        self.assertEqual(report['derived_ticket_count'], 3)
        self.assertEqual(len(report['tickets_sample']), 1)

    def test_static_interleaved_seed_requires_explicit_applicability(self):
        unit = self.inspect()
        ticket = next(unit.iter_tickets())
        with self.assertRaisesRegex(inspector.InspectionError, 'pinned236'):
            unit.reconstruct_interleaved_seed(ticket, binary_contract_sha256='0' * 64)
        without_flag = replace(unit, job=replace(unit.job, options=tuple(
            (key, value) for key, value in unit.job.options if key != '--force-interleave')))
        with self.assertRaisesRegex(inspector.InspectionError, 'force-interleave'):
            without_flag.reconstruct_interleaved_seed(ticket, binary_contract_sha256=inspector.INTERLEAVED_SEED_CONTRACT_SHA256)
        for invalid in (replace(ticket, globalticket=ticket.globalticket + 1),
                        replace(ticket, jidx=unit.job.jitter_end),
                        replace(ticket, prefix_states=()),
                        replace(ticket, cidx=True)):
            with self.subTest(ticket=invalid), self.assertRaises(inspector.InspectionError):
                unit.reconstruct_interleaved_seed(invalid, binary_contract_sha256=inspector.INTERLEAVED_SEED_CONTRACT_SHA256)
        result = unit.reconstruct_interleaved_seed(ticket, binary_contract_sha256=inspector.INTERLEAVED_SEED_CONTRACT_SHA256)
        self.assertEqual(result.schedule_index, 0)
        self.assertEqual(result.constructor_seed_u64, 0x3F3923A1FFE5274B)
        self.assertEqual(result.splitmix_words_u64, (0x1F6644FFE991528E, 0x88EA6A1724976E92,
                                                    0xA7FEC31A98C28069, 0xD2B0CEDC1B1CEC94))
        self.assertFalse(result.current_binary_verified)
        self.assertFalse(result.dispatcher_path_verified)
        self.assertFalse(result.full_dfs_compatibility_verified)
        changed_command_seed = replace(unit, job=replace(unit.job, seed=123))
        self.assertEqual(result, changed_command_seed.reconstruct_interleaved_seed(
            ticket, binary_contract_sha256=inspector.INTERLEAVED_SEED_CONTRACT_SHA256))

    def test_splitmix_static_vectors_include_uint64_wraparound(self):
        # Fixed vectors calculated independently using JavaScript BigInt and
        # modulo2**64 arithmetic, not this Python implementation's bit masks.
        vectors = ((0, 0, 0x6FECC3EC50DD6918,
                    (0x75856F745165F252, 0x8674BBC2735955AF, 0x5C1D49A70D26949A, 0x8CED152EF453EFD6)),
                   (99999, 2**32 - 1, 0x37AF98DE07958B20,
                    (0x8CD9097F43048540, 0x6724BF49D4EFB8FB, 0x0CFD543B02EB7FBB0, 0x9BC9A465EB5245C8)),
                   (2158, 251861254, 0x9EEFA1DF8D5F3B30,
                    (0x21B30C1FC8B3BD7C, 0x8BD22276C1892863, 0xF1BFF79230A2DF54, 0xA24DB033F41E0D51)))
        for cidx, jidx, expected_seed, expected_words in vectors:
            with self.subTest(cidx=cidx, jidx=jidx):
                seed, words = inspector._interleaved_constructor_state(cidx, jidx)
                self.assertEqual(seed, expected_seed)
                self.assertEqual(words, expected_words)
                self.assertTrue(all(0 <= word < 2**64 for word in words))

    def test_observed_shard_formula_matches_brute_force(self):
        job = self.inspect().job
        for nroots in range(1, 12):
            for shards in range(1, 12):
                for shard in range(shards):
                    trial = replace(job, shard_count=shards, shard_index=shard, jitter_start=7, jitter_end=19)
                    derived = sorted((j * nroots + c, j, c)
                                     for c, first, period, _ in inspector._ticket_ranges(trial, nroots, range(nroots))
                                     for j in range(first, trial.jitter_end, period))
                    brute = [(j * nroots + c, j, c) for j in range(7, 19) for c in range(nroots)
                             if ((j * nroots + c) - 7 * nroots) % shards == shard]
                    self.assertEqual(derived, brute)

    def test_live_soft_links_require_project_root_and_stay_inside_it(self):
        original = (self.slot / 'job.xml').read_bytes()
        (self.project / 'job_original.xml').write_bytes(original)
        self.write('job.xml', '<soft_link>../project/job_original.xml</soft_link>')
        with self.assertRaisesRegex(inspector.InspectionError, 'explicit project_root'):
            self.inspect()
        unit = self.inspect(project_root=self.project)
        self.assertEqual(unit.assets[0].source_name, 'job_original.xml')
        (self.root / 'outside.xml').write_bytes(original)
        self.write('job.xml', '<soft_link>../outside.xml</soft_link>')
        with self.assertRaisesRegex(inspector.InspectionError, 'inside project_root'):
            self.inspect(project_root=self.project)

    def test_soft_link_loop_and_xml_entities_rejected(self):
        (self.project / 'job_loop.xml').write_text('<soft_link>job_loop.xml</soft_link>', encoding='utf-8')
        self.write('job.xml', '<soft_link>../project/job_loop.xml</soft_link>')
        with self.assertRaisesRegex(inspector.InspectionError, 'Too many'):
            self.inspect(project_root=self.project)
        self.write('job.xml', '<!DOCTYPE job_desc [<!ENTITY x "unsafe">]>' + job_xml())
        with self.assertRaisesRegex(inspector.InspectionError, 'forbidden'):
            self.inspect()

    def test_original_project_catalog_and_allow_target_families(self):
        pairs = [('campaign_catalog.txt', 'catalog_sha' + hashlib.sha256(CATALOG.encode()).hexdigest() + '.txt'),
                 ('exploration_allow.txt', 'allow_eternity_cf_xp_test_s1_r10_14.txt')]
        for logical, source in pairs:
            (self.project / source).write_bytes((self.slot / logical).read_bytes())
            self.write(logical, '<soft_link>../project/' + source + '</soft_link>')
        self.assertEqual(self.inspect(project_root=self.project).ticket_count, 3)

    def test_unselected_sensitive_file_is_never_read(self):
        self.write('init_data.xml', 'THIS MUST NOT BE PARSED')
        self.assertEqual(self.inspect().ticket_count, 3)
        with self.assertRaisesRegex(inspector.InspectionError, 'Unsupported logical'):
            inspector._asset(self.slot, 'init_data.xml', None, 1000)
        (self.project / 'init_data.xml').write_text('SECRET DO NOT READ', encoding='utf-8')
        self.write('job.xml', '<soft_link>../project/init_data.xml</soft_link>')
        real = inspector._stable_bytes
        def guarded(path, limit):
            self.assertNotEqual(path.name, 'init_data.xml')
            return real(path, limit)
        with patch.object(inspector, '_stable_bytes', side_effect=guarded):
            with self.assertRaisesRegex(inspector.InspectionError, 'target filename'):
                self.inspect(project_root=self.project)

    def test_changing_inputs_rejected(self):
        real = inspector._asset
        calls = 0
        def change(*args, **kwargs):
            nonlocal calls
            value = real(*args, **kwargs)
            calls += 1
            if calls == 7:
                return replace(value, sha256='0' * 64)
            return value
        with patch.object(inspector, '_asset', side_effect=change):
            with self.assertRaisesRegex(inspector.InspectionError, 'changed during snapshot'):
                self.inspect()

    def test_oversized_input_rejected(self):
        self.write('job.xml', ' ' * (inspector.MAX_SMALL + 1))
        with self.assertRaisesRegex(inspector.InspectionError, 'exceeds'):
            self.inspect()

    def test_command_unknown_duplicate_and_missing_arguments_rejected(self):
        for command in (COMMAND + ' --execute anything', COMMAND + ' --seed 5', COMMAND.replace('--force-interleave ', ''),
                        COMMAND.replace('--root-allow exploration_allow.txt', '--root-allow ../../init_data.xml'),
                        COMMAND.replace('--force-jitters 14', '--force-jitters 10'),
                        COMMAND.replace('--force-jitter-start 10', '--force-jitter-start 3000000000'),
                        COMMAND + ' --main-fast 2', COMMAND + ' --exact-endgame 999:0',
                        COMMAND + ' --exact-endgame 244:0', COMMAND + ' --endgame-probe 21',
                        COMMAND.replace('--shard-index 1', '--shard-index 4')):
            with self.subTest(command=command), self.assertRaises(inspector.InspectionError):
                inspector.parse_job(job_xml(command).encode())

    def test_catalog_strict_pairs_duplicates_and_geometry(self):
        for data in ('1/0', '\n' + CATALOG, '# comment\n' + CATALOG,
                     CATALOG + CATALOG.splitlines()[0] + '\n', CATALOG.replace('26/0', '1/0', 1)):
            with self.subTest(data=data), self.assertRaises(inspector.InspectionError):
                inspector.parse_catalog(data.encode())
        self.write('campaign_catalog.txt', CATALOG.replace('26/0', '26/1', 1))
        with self.assertRaisesRegex(inspector.InspectionError, 'invalid frame|mismatched prefix'):
            self.inspect()

    def test_allow_indices_are_zero_based_unique_and_bounded(self):
        for value in ('', '0\n0', '-1', '3', '1 instruction'):
            with self.subTest(value=value), self.assertRaises(inspector.InspectionError):
                inspector.parse_allow(value.encode(), 3)

    def test_piece_and_hint_changes_rejected(self):
        original = (self.slot / '256pieces.txt').read_text()
        self.write('256pieces.txt', original.replace('1 0 0 17', '2 0 0 17', 1))
        with self.assertRaisesRegex(inspector.InspectionError, 'piece definitions differ'):
            self.inspect()
        self.write('256pieces.txt', original)
        self.write('campaign_hints.txt', HINTS.replace('249 2 2 2', '249 2 2 1'))
        with self.assertRaisesRegex(inspector.InspectionError, 'official five clues'):
            self.inspect()

    def test_campaign_mismatch_is_reported_not_invented_as_cpu_config_hash(self):
        first = self.inspect()
        self.manifest['expected']['root_catalog_sha256'] = '0' * 64
        self.write('campaign_manifest.json', json.dumps(self.manifest))
        second = self.inspect()
        self.assertEqual(dict(second.manifest_hash_checks)['campaign_catalog.txt'], 'mismatch-runtime-input')
        self.assertNotEqual(first.input_sha256, second.input_sha256)
        self.assertEqual(list(first.iter_tickets()), list(second.iter_tickets()))

    def test_journal_opaque_identity_is_retained_and_bound(self):
        folder = self.slot / 'ticketlog'; folder.mkdir()
        journal = {'format_version':2, 'record_size':40, 'endianness':'little',
                   'run_id':'0123456789abcdef', 'cfg_hash':'fedcba9876543210', 'build_id':'test build',
                   'epoch':'cloudflare-boinc-explore', 'shard_index':1, 'shard_count':4, 'nroots':3,
                   'force_jitters':14, 'force_interleave':1, 'node_cap':2000000000, 'seed':6152026,
                   'min_save':463, 'dedup_key':inspector.DEDUP_KEY}
        target = folder / 'seg_0123456789abcdef.manifest.json'
        target.write_text(json.dumps(journal), encoding='utf-8')
        identity = self.inspect().journals[0]
        self.assertEqual(identity.cfg_hash, 'fedcba9876543210')
        self.assertEqual(identity.dedup_key, inspector.DEDUP_KEY)
        journal['shard_index'] = 2
        target.write_text(json.dumps(journal), encoding='utf-8')
        with self.assertRaisesRegex(inspector.InspectionError, 'shard_index disagrees'):
            self.inspect()

    def test_json_duplicate_boolean_revision_and_frame_rejected(self):
        for text in ('{"campaign_revision":65,"campaign_revision":66}',
                     json.dumps({**self.manifest, 'campaign_revision':True}),
                     json.dumps({**self.manifest, 'frame_k':1})):
            with self.subTest(text=text):
                self.write('campaign_manifest.json', text)
                with self.assertRaises(inspector.InspectionError):
                    self.inspect()


if __name__ == '__main__':
    unittest.main()

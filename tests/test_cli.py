import io
import unittest
from unittest.mock import patch

from sltop import cli


class ParseMemToMbTests(unittest.TestCase):
    def test_parse_mem_to_mb(self) -> None:
        cases = {
            "30000M": 30000,
            "30G": 30720,
            "1T": 1048576,
            "": 0,
            "invalid": 0,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(cli.parse_mem_to_mb(value), expected)


class ParseReqTresTests(unittest.TestCase):
    def test_parse_req_tres_with_gres_gpu(self) -> None:
        raw = "JobId=101 ReqTRES=cpu=25,mem=30G,node=1,billing=25,gres/gpu=2"
        self.assertEqual(cli.parse_req_tres(raw), (25, 2, 30720))

    def test_parse_req_tres_with_gpu_fallback(self) -> None:
        raw = "JobId=103 ReqTRES=mem=16G,node=1,cpu=5,gpu=1"
        self.assertEqual(cli.parse_req_tres(raw), (5, 1, 16384))

    def test_parse_req_tres_missing(self) -> None:
        raw = "JobId=777 UserId=user1"
        self.assertEqual(cli.parse_req_tres(raw), (0, 0, 0))

    def test_parse_req_tres_sums_multiple_gpu_types(self) -> None:
        raw = "ReqTRES=cpu=4,mem=8G,gres/gpu:a100=1,gres/gpu:l40=2"
        self.assertEqual(cli.parse_req_tres(raw), (4, 3, 8192))

    def test_parse_req_tres_falls_back_to_alloctres_for_gpu(self) -> None:
        raw = (
            "JobId=200 ReqTRES=cpu=8,mem=30G,node=1,billing=8 "
            "AllocTRES=cpu=8,mem=30G,node=1,billing=8,gres/gpu=2"
        )
        self.assertEqual(cli.parse_req_tres(raw), (8, 2, 30720))

    def test_parse_req_tres_prefers_reqtres_over_tres_per_job(self) -> None:
        raw = (
            "JobId=205 ReqTRES=cpu=8,mem=30G,node=1,gres/gpu=1 "
            "TresPerJob=gres/gpu:2"
        )
        self.assertEqual(cli.parse_req_tres(raw), (8, 1, 30720))

    def test_parse_req_tres_falls_back_to_tres_per_job_for_gpu(self) -> None:
        raw = "JobId=206 ReqTRES=cpu=30,mem=30G,node=1,billing=30 TresPerJob=gres/gpu:2"
        self.assertEqual(cli.parse_req_tres(raw), (30, 2, 30720))

    def test_parse_req_tres_falls_back_to_tres_per_node_for_gpu(self) -> None:
        raw = "JobId=201 ReqTRES=cpu=4,mem=16G,node=1 TresPerNode=gres:gpu:2"
        self.assertEqual(cli.parse_req_tres(raw), (4, 2, 16384))

    def test_parse_req_tres_falls_back_to_req_gres_for_gpu(self) -> None:
        raw = "JobId=204 ReqTRES=cpu=4,mem=8G,node=1 ReqGRES=gpu:2"
        self.assertEqual(cli.parse_req_tres(raw), (4, 2, 8192))

    def test_parse_req_tres_falls_back_to_alloc_gres_for_gpu(self) -> None:
        raw = "JobId=207 ReqTRES=cpu=4,mem=8G,node=1 AllocGRES=gpu:a100:2"
        self.assertEqual(cli.parse_req_tres(raw), (4, 2, 8192))

    def test_parse_req_tres_falls_back_to_gres_for_gpu(self) -> None:
        raw = "JobId=202 ReqTRES=cpu=4,mem=8G,node=1 Gres=gpu:a100:2(S:0-1)"
        self.assertEqual(cli.parse_req_tres(raw), (4, 2, 8192))

    def test_parse_req_tres_falls_back_to_tres_per_task_for_gpu(self) -> None:
        raw = "JobId=203 ReqTRES=cpu=4,mem=8G,node=1 TresPerTask=gres:gpu:1 NumTasks=2"
        self.assertEqual(cli.parse_req_tres(raw), (4, 2, 8192))

    def test_parse_req_tres_reproduces_real_tres_per_job_case(self) -> None:
        raw = (
            "JobId=30 JobState=RUNNING NumTasks=2 ReqTRES=cpu=30,mem=30G,node=1,"
            "billing=30 AllocTRES=cpu=30,mem=30G,node=1,billing=30 "
            "TresPerJob=gres/gpu:2 TresPerTask=cpu=15"
        )
        self.assertEqual(cli.parse_req_tres(raw), (30, 2, 30720))


class JobsAndAggregateTests(unittest.TestCase):
    def test_get_jobs_for_node_filters_and_maps_states(self) -> None:
        output = "\n".join(
            [
                "101|user1|RUNNING",
                "110|user1|PENDING",
                "oops|user2|RUNNING",
                "120|user3|COMPLETED",
                "",
            ]
        )
        with patch("sltop.cli.run_command", return_value=output) as mocked_run:
            jobs = cli.get_jobs_for_node("node-a", "RUNNING")

        self.assertEqual(jobs, [(101, "user1", "RUN")])
        mocked_run.assert_called_once_with(
            ["squeue", "-h", "-w", "node-a", "-t", "RUNNING", "-o", "%A|%u|%T"]
        )

    def test_get_jobs_for_node_without_node_filter(self) -> None:
        output = "\n".join(
            [
                "110|user1|PENDING",
                "",
            ]
        )
        with patch("sltop.cli.run_command", return_value=output) as mocked_run:
            jobs = cli.get_jobs_for_node(None, "PENDING")

        self.assertEqual(jobs, [(110, "user1", "PEND")])
        mocked_run.assert_called_once_with(
            ["squeue", "-h", "-t", "PENDING", "-o", "%A|%u|%T"]
        )

    def test_build_user_aggregates(self) -> None:
        resources = [
            cli.JobResource(
                job_id=102, user="user1", state="RUN", cpu=5, gpu=1, mem_mb=4096
            ),
            cli.JobResource(
                job_id=101, user="user1", state="RUN", cpu=20, gpu=1, mem_mb=26624
            ),
            cli.JobResource(
                job_id=110, user="user1", state="PEND", cpu=10, gpu=1, mem_mb=30720
            ),
            cli.JobResource(
                job_id=103, user="user2", state="RUN", cpu=5, gpu=1, mem_mb=16384
            ),
        ]
        aggregates = cli.build_user_aggregates(resources)

        self.assertEqual(set(aggregates.keys()), {"user1", "user2"})
        self.assertEqual(aggregates["user1"].run.cpu_total, 25)
        self.assertEqual(aggregates["user1"].run.gpu_total, 2)
        self.assertEqual(aggregates["user1"].run.mem_mb_total, 30720)
        self.assertEqual(aggregates["user1"].run.job_ids, [101, 102])
        self.assertEqual(aggregates["user1"].pend.cpu_total, 10)
        self.assertEqual(aggregates["user1"].pend.gpu_total, 1)
        self.assertEqual(aggregates["user1"].pend.mem_mb_total, 30720)
        self.assertEqual(aggregates["user1"].pend.job_ids, [110])

        self.assertEqual(aggregates["user2"].run.cpu_total, 5)
        self.assertEqual(aggregates["user2"].run.gpu_total, 1)
        self.assertEqual(aggregates["user2"].run.mem_mb_total, 16384)
        self.assertEqual(aggregates["user2"].run.job_ids, [103])
        self.assertEqual(aggregates["user2"].pend.cpu_total, 0)
        self.assertEqual(aggregates["user2"].pend.gpu_total, 0)
        self.assertEqual(aggregates["user2"].pend.mem_mb_total, 0)
        self.assertEqual(aggregates["user2"].pend.job_ids, [])

    def test_print_users_report(self) -> None:
        resources = [
            cli.JobResource(
                job_id=101, user="user1", state="RUN", cpu=25, gpu=2, mem_mb=30720
            ),
            cli.JobResource(
                job_id=102, user="user1", state="RUN", cpu=0, gpu=0, mem_mb=0
            ),
            cli.JobResource(
                job_id=110, user="user1", state="PEND", cpu=10, gpu=1, mem_mb=30720
            ),
            cli.JobResource(
                job_id=103, user="user2", state="RUN", cpu=5, gpu=1, mem_mb=16384
            ),
        ]
        aggregates = cli.build_user_aggregates(resources)

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            cli.print_users_report(aggregates)

        output = stdout.getvalue()
        self.assertIn("USERS :", output)
        self.assertIn("user1  RUN", output)
        self.assertIn("jobs: 101,102", output)
        self.assertIn("PEND cpu 10  gpu 1  mem 30720MB   jobs: 110", output)
        self.assertIn("user2  RUN", output)
        self.assertIn("PEND cpu  0  gpu 0  mem     0MB   jobs: -", output)

    def test_get_job_resources_skips_missing_job_error(self) -> None:
        run_jobs = [(101, "user1", "RUN"), (103, "user2", "RUN")]
        pending_jobs = [(110, "user1", "PEND"), (999, "user3", "PEND")]

        with patch(
            "sltop.cli.get_jobs_for_node",
            side_effect=[run_jobs, pending_jobs],
        ):
            with patch(
                "sltop.cli.get_job_resource",
                side_effect=[
                    cli.JobResource(
                        job_id=101,
                        user="user1",
                        state="RUN",
                        cpu=25,
                        gpu=2,
                        mem_mb=30720,
                    ),
                    cli.JobResource(
                        job_id=103,
                        user="user2",
                        state="RUN",
                        cpu=5,
                        gpu=1,
                        mem_mb=16384,
                    ),
                    RuntimeError(
                        "failed to run scontrol show job -o 110: "
                        "slurm_load_jobs error: Invalid job id specified"
                    ),
                ],
            ):
                resources = cli.get_job_resources("node-a")

        self.assertEqual([r.job_id for r in resources], [101, 103])

    def test_get_job_resources_raises_non_missing_error(self) -> None:
        run_jobs = [(101, "user1", "RUN")]
        with patch(
            "sltop.cli.get_jobs_for_node",
            side_effect=[run_jobs, []],
        ):
            with patch(
                "sltop.cli.get_job_resource",
                side_effect=RuntimeError(
                    "failed to run scontrol show job -o 101: permission denied"
                ),
            ):
                with self.assertRaises(RuntimeError):
                    cli.get_job_resources("node-a")

    def test_get_job_resources_includes_pending_for_run_users(self) -> None:
        run_jobs = [(31, "kurita", "RUN")]
        pending_jobs = [(32, "kurita", "PEND"), (40, "other", "PEND")]

        with patch(
            "sltop.cli.get_jobs_for_node",
            side_effect=[run_jobs, pending_jobs],
        ):
            with patch(
                "sltop.cli.get_job_resource",
                side_effect=[
                    cli.JobResource(
                        job_id=31,
                        user="kurita",
                        state="RUN",
                        cpu=30,
                        gpu=2,
                        mem_mb=30720,
                    ),
                    cli.JobResource(
                        job_id=32,
                        user="kurita",
                        state="PEND",
                        cpu=30,
                        gpu=2,
                        mem_mb=30720,
                    ),
                ],
            ):
                resources = cli.get_job_resources("node-a")

        self.assertEqual([r.job_id for r in resources], [31, 32])


if __name__ == "__main__":
    unittest.main()

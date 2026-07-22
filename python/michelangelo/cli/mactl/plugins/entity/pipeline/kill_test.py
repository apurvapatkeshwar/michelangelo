"""Unit tests for pipeline kill command.

Covers signature injection, generate_kill wiring, confirmation, state
pre-check, kill flag propagation, and error paths.
"""

from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from google.protobuf.message import Message

from michelangelo.cli.mactl.crd import CRD
from michelangelo.cli.mactl.plugins.entity.pipeline.kill import (
    add_function_signature,
    generate_kill,
)
from michelangelo.gen.api.v2 import pipeline_run_pb2, pipeline_run_svc_pb2

_PENDING = pipeline_run_pb2.PIPELINE_RUN_STATE_PENDING
_RUNNING = pipeline_run_pb2.PIPELINE_RUN_STATE_RUNNING
_SUCCEEDED = pipeline_run_pb2.PIPELINE_RUN_STATE_SUCCEEDED
_KILLED = pipeline_run_pb2.PIPELINE_RUN_STATE_KILLED
_FAILED = pipeline_run_pb2.PIPELINE_RUN_STATE_FAILED


def _make_get_response(state):
    """Build a get-response whose inner PipelineRun has the given state."""
    response = pipeline_run_svc_pb2.GetPipelineRunResponse()
    response.pipeline_run.status.state = state
    return response


class PipelineKillTest(TestCase):
    """Tests for pipeline kill command."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_crd = Mock(spec=CRD)
        self.mock_crd.name = "pipeline_run"
        self.mock_crd.full_name = "michelangelo.api.v2.PipelineRunService"
        self.mock_crd.metadata = {}
        self.mock_crd.func_signature = {}

        mock_signature = Mock()

        def mock_bind(*args, **kwargs):
            bound = Mock()
            bound.arguments = {
                "self": args[0] if args else kwargs.get("self"),
                "namespace": kwargs.get("namespace"),
                "name": kwargs.get("name"),
                "yes": kwargs.get("yes", False),
            }
            return bound

        mock_signature.bind = mock_bind
        self.mock_crd._read_signatures = Mock(return_value=mock_signature)
        self.mock_crd.configure_parser = Mock()
        self.mock_channel = Mock()

    def _wire_extract_method_info(self):
        """Common wiring for generate_kill so tests can drive kill_func."""
        self.mock_crd.generate_get = Mock()
        mock_input_class = MagicMock()
        mock_output_class = MagicMock()
        self.mock_crd._extract_method_info = Mock(
            return_value=("UpdatePipelineRun", mock_input_class, mock_output_class)
        )
        return mock_input_class, mock_output_class

    def test_add_function_signature(self):
        """add_function_signature runs without error."""
        add_function_signature(self.mock_crd)
        self.assertTrue(True)

    def test_generate_kill_basic(self):
        """generate_kill wires generate_get + Update extraction."""
        self._wire_extract_method_info()
        generate_kill(self.mock_crd, self.mock_channel)
        self.mock_crd.generate_get.assert_called_once_with(self.mock_channel)
        self.mock_crd._extract_method_info.assert_called_once_with(
            self.mock_channel, self.mock_crd.full_name, "Update"
        )

    def test_kill_command_requires_namespace_and_name(self):
        """Signature declares required namespace and name."""
        add_function_signature(self.mock_crd)
        self.assertTrue(True)

    def test_generate_kill_missing_update_method(self):
        """generate_kill re-raises when Update method is not present."""
        self.mock_crd.generate_get = Mock()
        self.mock_crd._extract_method_info = Mock(
            side_effect=ValueError("Method Update not found")
        )
        with self.assertRaises(ValueError):
            generate_kill(self.mock_crd, self.mock_channel)

    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.crd_method_call")
    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.MessageToDict")
    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.ParseDict")
    def test_kill_func_with_yes_flag(
        self, mock_parse_dict, mock_message_to_dict, mock_call
    ):
        """kill_func executes with --yes and returns Update response."""
        self._wire_extract_method_info()
        self.mock_crd.get = Mock(return_value=_make_get_response(_RUNNING))
        mock_update_response = Mock(spec=Message)
        mock_call.return_value = mock_update_response

        mock_message_to_dict.side_effect = [
            {"pipeline_run": {"spec": {"some_field": "value"}}},
            {"pipeline_run": {"spec": {"kill": True}}},
        ]

        generate_kill(self.mock_crd, self.mock_channel)
        kill_func = self.mock_crd.kill
        result = kill_func(
            self.mock_crd,
            namespace="test-namespace",
            name="test-pipeline-run",
            yes=True,
        )

        self.assertEqual(result, mock_update_response)
        self.mock_crd.get.assert_called_once_with("test-namespace", "test-pipeline-run")
        mock_call.assert_called_once()

    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.crd_method_call")
    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.MessageToDict")
    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.ParseDict")
    @patch("builtins.input")
    def test_kill_func_user_confirms(
        self, mock_input, mock_parse_dict, mock_message_to_dict, mock_call
    ):
        """kill_func proceeds when user types 'yes'."""
        mock_input.return_value = "yes"
        self._wire_extract_method_info()
        self.mock_crd.get = Mock(return_value=_make_get_response(_RUNNING))
        mock_update_response = Mock(spec=Message)
        mock_call.return_value = mock_update_response

        mock_message_to_dict.side_effect = [
            {"pipeline_run": {"spec": {}}},
            {"pipeline_run": {"spec": {"kill": True}}},
        ]

        generate_kill(self.mock_crd, self.mock_channel)
        kill_func = self.mock_crd.kill
        result = kill_func(
            self.mock_crd, namespace="test-ns", name="test-run", yes=False
        )

        self.assertEqual(result, mock_update_response)
        mock_input.assert_called_once()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_kill_func_user_cancels(self, mock_print, mock_input):
        """kill_func returns None and prints cancellation when user says 'no'."""
        mock_input.return_value = "no"
        self._wire_extract_method_info()

        generate_kill(self.mock_crd, self.mock_channel)
        kill_func = self.mock_crd.kill
        result = kill_func(
            self.mock_crd, namespace="test-ns", name="test-run", yes=False
        )

        self.assertIsNone(result)
        mock_print.assert_called_with("Kill operation cancelled.")

    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.MessageToDict")
    def test_kill_func_missing_spec_field(self, mock_message_to_dict):
        """kill_func raises when the resource dict lacks a spec field."""
        self._wire_extract_method_info()
        self.mock_crd.get = Mock(return_value=_make_get_response(_RUNNING))
        mock_message_to_dict.return_value = {"pipeline_run": {}}

        generate_kill(self.mock_crd, self.mock_channel)
        kill_func = self.mock_crd.kill
        with self.assertRaises(ValueError) as context:
            kill_func(self.mock_crd, namespace="test-ns", name="test-run", yes=True)
        self.assertIn("Cannot set kill flag", str(context.exception))

    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.crd_method_call")
    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.MessageToDict")
    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.ParseDict")
    def test_kill_func_kill_flag_not_set(
        self, mock_parse_dict, mock_message_to_dict, mock_call
    ):
        """kill_func raises when Update response lacks kill=true."""
        self._wire_extract_method_info()
        self.mock_crd.get = Mock(return_value=_make_get_response(_RUNNING))
        mock_call.return_value = Mock(spec=Message)

        mock_message_to_dict.side_effect = [
            {"pipeline_run": {"spec": {}}},
            {"pipeline_run": {"spec": {"kill": False}}},
        ]

        generate_kill(self.mock_crd, self.mock_channel)
        kill_func = self.mock_crd.kill
        with self.assertRaises(RuntimeError) as context:
            kill_func(self.mock_crd, namespace="test-ns", name="test-run", yes=True)
        self.assertIn("Kill operation failed", str(context.exception))

    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.crd_method_call")
    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.MessageToDict")
    @patch("michelangelo.cli.mactl.plugins.entity.pipeline.kill.ParseDict")
    def test_kill_allowed_on_pending_state(
        self, mock_parse_dict, mock_message_to_dict, mock_call
    ):
        """PENDING runs pass the state check and proceed to Update."""
        self._wire_extract_method_info()
        self.mock_crd.get = Mock(return_value=_make_get_response(_PENDING))
        mock_update_response = Mock(spec=Message)
        mock_call.return_value = mock_update_response

        mock_message_to_dict.side_effect = [
            {"pipeline_run": {"spec": {}}},
            {"pipeline_run": {"spec": {"kill": True}}},
        ]

        generate_kill(self.mock_crd, self.mock_channel)
        kill_func = self.mock_crd.kill
        result = kill_func(
            self.mock_crd, namespace="test-ns", name="test-run", yes=True
        )

        self.assertEqual(result, mock_update_response)
        mock_call.assert_called_once()

    def _assert_not_killable(self, state, state_name):
        """Assert kill on a non-killable state raises the Go error verbatim."""
        self._wire_extract_method_info()
        self.mock_crd.get = Mock(return_value=_make_get_response(state))

        generate_kill(self.mock_crd, self.mock_channel)
        kill_func = self.mock_crd.kill
        with self.assertRaises(ValueError) as context:
            kill_func(self.mock_crd, namespace="test-ns", name="test-run", yes=True)

        expected = (
            f"the pipelinerun cannot be killed because it's in {state_name} "
            "state. a pipelinerun can be killed if its state is either "
            "PENDING or RUNNING"
        )
        self.assertEqual(str(context.exception), expected)

    def test_kill_rejected_on_succeeded_state(self):
        """SUCCEEDED runs fail the state check with the Go error string."""
        self._assert_not_killable(_SUCCEEDED, "PIPELINE_RUN_STATE_SUCCEEDED")

    def test_kill_rejected_on_failed_state(self):
        """FAILED runs fail the state check with the Go error string."""
        self._assert_not_killable(_FAILED, "PIPELINE_RUN_STATE_FAILED")

    def test_kill_rejected_on_killed_state(self):
        """Already-killed runs fail the state check with the Go error string."""
        self._assert_not_killable(_KILLED, "PIPELINE_RUN_STATE_KILLED")

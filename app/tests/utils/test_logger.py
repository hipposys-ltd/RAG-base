import pytest

from app.utils.logger import Logger


class TestLogger:
    """Test the Logger class."""

    def teardown_method(self):
        # Reinitialize the logger with the default configuration.
        Logger(force_recreate=True)

    @pytest.mark.parametrize('config_to_use,expected_app_name', [

        # Default configuration, no changes.
        ({}, 'RAG-App'),

        # Change app name.
        ({'name': 'RAGRAGRAG123'}, 'RAGRAGRAG123'),
    ])
    def test_logger_config(self, caplog, config_to_use: dict, expected_app_name: str):
        """Test that different configurations affect logging output."""

        # Setup
        logger = Logger(config_to_use=config_to_use, force_recreate=True)

        with caplog.at_level('INFO'):
            # Run
            logger.get_logger().info('Test message')

        # Validate
        assert len(caplog.text.splitlines()) == 1
        assert 'Test message' in caplog.text
        assert expected_app_name in caplog.text

    @pytest.mark.parametrize('log_level,expected_outputs', [

        # Log level: DEBUG
        ('DEBUG', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
        
        # Log level: INFO
        ('INFO', ['INFO', 'WARNING', 'ERROR', 'CRITICAL']),

        # Log level: WARNING
        ('WARNING', ['WARNING', 'ERROR', 'CRITICAL']),

        # Log level: ERROR
        ('ERROR', ['ERROR', 'CRITICAL']),

        # Log level: CRITICAL
        ('CRITICAL', ['CRITICAL']),
    ])
    def test_logging_level(self, caplog, log_level: str, expected_outputs: list[str]):
        """Test that different logging levels affect logging output."""
        
        # Setup
        logger = Logger(config_to_use={'level': log_level}, force_recreate=True)

        # Run
        logger.get_logger().debug('-Test-Print-Debug-')
        logger.get_logger().info('-Test-Print-Info-')
        logger.get_logger().warning('-Test-Print-Warning-')
        logger.get_logger().error('-Test-Print-Error-')
        logger.get_logger().critical('-Test-Print-Critical-')

        # Validate
        outputs = caplog.text.splitlines()
        assert len(outputs) == len(expected_outputs)
        for output, expected_output in zip(outputs, expected_outputs):
            assert f'-Test-Print-{expected_output.capitalize()}-' in output

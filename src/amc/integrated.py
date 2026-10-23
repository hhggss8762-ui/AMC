"""Extend the unchanged signal-detector 0.2.1 application inside its image."""
from app.main import create_application

from .service import install_amc


app = create_application()
app.title = "Signal detection and modulation recognition"
app.description = (
    "Original signal-detector 0.2.1 endpoints and an independent prepared-IQ "
    "batch classification API. The receiver performs IQ extraction and DDC."
)
app.version = "0.1.0"
install_amc(app)

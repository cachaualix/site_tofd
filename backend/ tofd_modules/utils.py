import io
import matplotlib.pyplot as plt

def fig_to_png(fig) -> bytes:
    """Convertit une figure matplotlib en PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def fig_to_bytes(fig) -> bytes:
    """Alias de fig_to_png."""
    return fig_to_png(fig)

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =============================================================================
# LOAD B-SCAN
# =============================================================================

def load_bscan_grayscale(
    path: Union[str, Path]
) -> np.ndarray:

    img = Image.open(path).convert("L")  # Open the image and convert it to grayscale.
    return np.asarray(img)  # Convert the PIL image into a NumPy array.


# =============================================================================
# OPTIONAL CROP
# =============================================================================

def crop_bscan(
    bscan_gray: np.ndarray,
    row_range: Optional[Sequence[int]] = None,
    col_range: Optional[Sequence[int]] = None,
) -> np.ndarray:

    r0, r1 = row_range if row_range is not None else (0, bscan_gray.shape[0])  # Define row limits.
    c0, c1 = col_range if col_range is not None else (0, bscan_gray.shape[1])  # Define column limits.

    return bscan_gray[r0:r1, c0:c1]  # Return the selected part of the B-scan.


# =============================================================================
# EXTRACT A-SCANS
# =============================================================================

def extract_columns(
    bscan_gray: np.ndarray,
    normalize: bool = True,
    assume_diverging_grey: bool = False, 
) -> np.ndarray:

    bscan_float = bscan_gray.astype(np.float32)  # Convert pixels to floating-point values.

    if assume_diverging_grey:
        signal_matrix = (bscan_float - 127.5) / 127.5  # Convert 0-255 grey levels to approximately -1 to +1.

    elif normalize:
        signal_matrix = bscan_float / 255.0  # Normalize grey levels between 0 and 1.

    else:
        signal_matrix = bscan_float  # Keep the original grey levels.

    columns = signal_matrix.T  # Transpose so that each row corresponds to one A-scan.
    columns = np.ascontiguousarray(columns)  # Store the array contiguously for faster repeated access.

    return columns

# =============================================================================
# SINGLE A-SCAN
# =============================================================================

def plot_column_profile(
    columns: np.ndarray,
    column_index: int,
    depth_axis: Optional[np.ndarray] = None,
    ax=None,
    title: Optional[str] = None,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))  # Create a figure if no axis was provided.

    signal = columns[column_index]  # Select the requested A-scan.

    y_axis = depth_axis if depth_axis is not None else np.arange(len(signal))  # Select the vertical axis.

    ax.plot(signal, y_axis, color="black")  # Plot the A-scan.

    ax.invert_yaxis()  # Put increasing time/depth downwards.

    ax.set_xlabel(
        "Grey intensity" if depth_axis is None else "Amplitude"
    )  # Set the x-axis label.

    ax.set_ylabel(
        "Row index" if depth_axis is None else "Time of flight (µs)"
    )  # Set the y-axis label.

    ax.set_title(title or f"Column {column_index}")  # Set the title.

    return ax


# =============================================================================
# MULTIPLE A-SCANS
# =============================================================================

def plot_multiple_column_profiles(
    columns: np.ndarray,
    column_indices: Sequence[int],
    depth_axis: Optional[np.ndarray] = None,
    ax=None,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))  # Create a figure if no axis was provided.

    y_axis = (
        depth_axis
        if depth_axis is not None
        else np.arange(columns.shape[1])
    )  # Select the vertical axis.

    for idx in column_indices:
        ax.plot(columns[idx], y_axis)  # Plot each selected A-scan.

    ax.invert_yaxis()  # Put increasing time/depth downwards.

    ax.set_xlabel(
        "Grey intensity" if depth_axis is None else "Amplitude"
    )  # Set the x-axis label.

    ax.set_ylabel(
        "Row index" if depth_axis is None else "Time of flight (µs)"
    )  # Set the y-axis label.

    ax.set_title("Overlaid column intensity profiles")  # Set the graph title.

    return ax


# =============================================================================
# B-SCAN + A-SCAN
# =============================================================================

def plot_bscan_with_selected_column(
    bscan_gray: np.ndarray,
    columns: np.ndarray,
    column_index: int,
    depth_axis: Optional[np.ndarray] = None,
    position_axis: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 6),
        gridspec_kw={"width_ratios": [1, 3]}
    )  # Create the two-panel figure.

    plot_column_profile(
        columns,
        column_index,
        depth_axis=depth_axis,
        ax=axes[0],
        title=f"A-scan at column {column_index}"
    )  # Plot the selected A-scan.

    extent = None  # Use default pixel coordinates unless physical axes are provided.

    if position_axis is not None and depth_axis is not None:
        extent = [
            position_axis[0],
            position_axis[-1],
            depth_axis[-1],
            depth_axis[0]
        ]  # Define physical coordinates for the image.

    axes[1].imshow(
        bscan_gray,
        cmap="gray",
        aspect="auto",
        extent=extent
    )  # Display the B-scan.

    x_marker = (
        position_axis[column_index]
        if position_axis is not None
        else column_index
    )  # Find the x-position of the selected column.

    axes[1].axvline(
        x_marker,
        color="red",
        linewidth=1.5
    )  # Draw the selected-column marker.

    axes[1].set_title("B-scan")  # Set the B-scan title.

    axes[1].set_xlabel(
        "Scanning distance" if position_axis is not None else "Column index"
    )  # Set the x-axis label.

    axes[1].set_ylabel(
        "Time of flight (µs)" if depth_axis is not None else "Row index"
    )  # Set the y-axis label.

    plt.tight_layout()  # Adjust the layout.

    if save_path:
        plt.savefig(save_path, dpi=150)  # Save the figure.
        print(f"Figure saved: {save_path}")  # Confirm the saved file.

    return fig, axes


# =============================================================================
# OPTIMIZED INTERACTIVE VIEWER
# =============================================================================

def interactive_ascan_viewer(
    bscan_gray: np.ndarray,
    columns: np.ndarray,
    depth_axis: Optional[np.ndarray] = None,
    position_axis: Optional[np.ndarray] = None,
    history_length: int = 10,
):

    n_columns, n_samples = columns.shape  # Get the number of A-scans and samples.

    print()
    print("=" * 70)
    print("MODE INTERACTIF OPTIMISÉ")
    print("=" * 70)
    print()
    print("Souris :")
    print("  - clic sur le B-scan")
    print("  - clic + déplacement = déplacement continu")
    print()
    print("Clavier :")
    print("  ← / → : colonne précédente / suivante")
    print("  Home  : première colonne")
    print("  End   : dernière colonne")
    print()
    print("=" * 70)

    if depth_axis is not None:
        y_axis = np.asarray(depth_axis, dtype=np.float32)  # Use the physical time axis.

    else:
        y_axis = np.arange(n_samples, dtype=np.float32)  # Use sample indices.

    if position_axis is not None:
        x_axis = np.asarray(position_axis, dtype=np.float32)  # Use the physical position axis.

    else:
        x_axis = np.arange(n_columns, dtype=np.float32)  # Use column indices.

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        gridspec_kw={"width_ratios": [3, 1]}
    )  # Create the B-scan and A-scan panels.

    ax_bscan = axes[0]  # Store the B-scan axis.
    ax_ascan = axes[1]  # Store the A-scan axis.

    try:
        fig.canvas.manager.set_window_title(
            "B-scan / A-scan - mode interactif"
        )  # Set the window title.

    except Exception:
        pass  # Ignore this if the backend does not support window titles.

    extent = [
        float(x_axis[0]),
        float(x_axis[-1]),
        float(y_axis[-1]),
        float(y_axis[0])
    ]  # Define the physical extent of the B-scan.

    ax_bscan.imshow(
        bscan_gray,
        cmap="gray",
        aspect="auto",
        extent=extent,
        interpolation="nearest"
    )  # Display the B-scan once.

    ax_bscan.set_xlabel(
        "Scanning position (mm)" if position_axis is not None else "Column index"
    )  # Set the B-scan x-axis label.

    ax_bscan.set_ylabel(
        "Time of flight (µs)" if depth_axis is not None else "Row index"
    )  # Set the B-scan y-axis label.

    vertical_line = ax_bscan.axvline(
        x_axis[0],
        color="red",
        linewidth=2.5,
        animated=True
    )  # Create the single movable red line.

    ax_ascan.set_xlabel(
        "Amplitude" if depth_axis is not None else "Grey intensity"
    )  # Set the A-scan x-axis label.

    ax_ascan.set_ylabel(
        "Time of flight (µs)" if depth_axis is not None else "Row index"
    )  # Set the A-scan y-axis label.

    ax_ascan.invert_yaxis()  # Put increasing time/depth downwards.

    signal_min = float(np.min(columns))  # Find the minimum signal value.
    signal_max = float(np.max(columns))  # Find the maximum signal value.

    if signal_min == signal_max:
        margin = 1.0  # Use a fixed margin for a constant signal.

    else:
        margin = 0.05 * (signal_max - signal_min)  # Add 5% margin around the signals.

    ax_ascan.set_xlim(
        signal_min - margin,
        signal_max + margin
    )  # Set the horizontal limits.

    ax_ascan.set_ylim(
        y_axis[-1],
        y_axis[0]
    )  # Set the vertical limits.

    colors = plt.cm.tab10(
        np.linspace(0, 1, history_length)
    )  # Generate different colors for previous A-scans.

    history_lines = []  # Store the previous A-scan curves.

    for i in range(history_length):
        line, = ax_ascan.plot(
            [],
            [],
            color=colors[i],
            linewidth=1.8,
            alpha=0.0,
            visible=False,
            animated=True
        )  # Create an initially invisible history curve.

        history_lines.append(line)  # Store the curve.

    current_line, = ax_ascan.plot(
        columns[0],
        y_axis,
        color="black",
        linewidth=2.5,
        animated=True,
        zorder=30
    )  # Create the current A-scan curve.

    current_index = 0  # Start at the first column.
    dragging = False  # Store whether the mouse is currently dragging.
    last_drawn_index = -1  # Force the first redraw.

    bscan_title = ax_bscan.set_title(
        f"B-scan — colonne 1/{n_columns}"
    )  # Create the B-scan title as a movable text artist.

    ascan_title = ax_ascan.set_title(
        f"A-scan — colonne 1/{n_columns}"
    )  # Create the A-scan title as a movable text artist.

    bscan_title.set_animated(True)  # Allow the B-scan title to be redrawn with blitting.
    ascan_title.set_animated(True)  # Allow the A-scan title to be redrawn with blitting.

    fig.canvas.draw()  # Perform the initial complete rendering.

    background = fig.canvas.copy_from_bbox(
        fig.bbox
    )  # Save the static background for fast redrawing.

    def redraw(index):

        nonlocal background
        nonlocal last_drawn_index

        if index == last_drawn_index:
            return  # Do nothing if the selected column has not changed.

        last_drawn_index = index  # Store the new selected column.

        current_line.set_data(
            columns[index],
            y_axis
        )  # Update the current A-scan.

        for line in history_lines:
            line.set_visible(False)  # Hide all previous A-scans before updating them.

        number_previous = min(
            history_length,
            index
        )  # Determine how many previous A-scans are available.

        for age in range(1, number_previous + 1):

            previous_index = index - age  # Find the corresponding previous column.
            line = history_lines[age - 1]  # Select the corresponding history curve.

            line.set_data(
                columns[previous_index],
                y_axis
            )  # Update the history curve with the previous A-scan.

            line.set_color(
                colors[age - 1]
            )  # Give this previous A-scan its own color.

            alpha = max(
                0.18,
                1.0 - 0.075 * age
            )  # Make older A-scans progressively more transparent.

            line.set_alpha(alpha)  # Apply the transparency.

            line.set_visible(True)  # Display this previous A-scan.

        vertical_line.set_xdata(
            [x_axis[index], x_axis[index]]
        )  # Move the red line to the selected column.

        bscan_title.set_text(
            f"B-scan — colonne {index + 1}/{n_columns}"
        )  # Update the B-scan title with the current column.

        ascan_title.set_text(
            f"A-scan — colonne {index + 1}/{n_columns}"
        )  # Update the A-scan title with the current column.

        fig.canvas.restore_region(
            background
        )  # Restore the unchanged background.

        for line in history_lines:
            if line.get_visible():
                ax_ascan.draw_artist(line)  # Draw only visible history curves.

        ax_ascan.draw_artist(
            current_line
        )  # Draw the current A-scan.

        ax_bscan.draw_artist(
            vertical_line
        )  # Draw the red column marker.

        ax_bscan.draw_artist(
            bscan_title
        )  # Redraw the updated B-scan title.

        ax_ascan.draw_artist(
            ascan_title
        )  # Redraw the updated A-scan title.

        fig.canvas.blit(
            fig.bbox
        )  # Update the window without redrawing the whole figure.

        fig.canvas.flush_events()  # Immediately display the changes.

    def mouse_to_column(event):

        if event.xdata is None:
            return None  # Ignore events outside the B-scan x-axis.

        x = event.xdata  # Get the mouse x-position.

        index = np.searchsorted(
            x_axis,
            x
        )  # Quickly find the closest position in the column axis.

        if index <= 0:
            return 0  # Clamp to the first column.

        if index >= n_columns:
            return n_columns - 1  # Clamp to the last column.

        if abs(x_axis[index] - x) < abs(x_axis[index - 1] - x):
            return index  # Select the right column if it is closer.

        return index - 1  # Otherwise select the left column.

    def on_press(event):

        nonlocal dragging
        nonlocal current_index

        if event.inaxes != ax_bscan:
            return  # Ignore clicks outside the B-scan.

        if event.button != 1:
            return  # Only respond to the left mouse button.

        dragging = True  # Start dragging.

        index = mouse_to_column(event)  # Convert the mouse position into a column.

        if index is not None:
            current_index = index  # Store the selected column.
            redraw(current_index)  # Update the display.

    def on_motion(event):

        nonlocal current_index

        if not dragging:
            return  # Do nothing if the mouse button is not held.

        if event.inaxes != ax_bscan:
            return  # Ignore movement outside the B-scan.

        index = mouse_to_column(event)  # Convert the mouse position into a column.

        if index is None:
            return  # Ignore invalid positions.

        if index == current_index:
            return  # Avoid unnecessary redraws.

        current_index = index  # Store the new column.
        redraw(current_index)  # Update the display.

    def on_release(event):

        nonlocal dragging

        if event.button == 1:
            dragging = False  # Stop dragging when the left button is released.

    def on_key(event):

        nonlocal current_index

        if event.key == "right":
            new_index = min(
                current_index + 1,
                n_columns - 1
            )  # Move one column to the right.

        elif event.key == "left":
            new_index = max(
                current_index - 1,
                0
            )  # Move one column to the left.

        elif event.key == "home":
            new_index = 0  # Go to the first column.

        elif event.key == "end":
            new_index = n_columns - 1  # Go to the last column.

        else:
            return  # Ignore all other keys.

        if new_index != current_index:
            current_index = new_index  # Store the new column.
            redraw(current_index)  # Update the display.

    fig.canvas.mpl_connect(
        "button_press_event",
        on_press
    )  # Connect mouse click events.

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_motion
    )  # Connect mouse movement events.

    fig.canvas.mpl_connect(
        "button_release_event",
        on_release
    )  # Connect mouse release events.

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key
    )  # Connect keyboard events.

    redraw(0)  # Display the first column.

    plt.show()  # Open the interactive window.

    return fig  # Return the figure.


# =============================================================================
# EXPORT
# =============================================================================

def save_columns_as_ascan_dataset(
    columns: np.ndarray,
    filepath: str
):

    np.save(filepath, columns)  # Save all extracted A-scans as a NumPy file.

    print(
        f"Saved {columns.shape[0]} extracted A-scans "
        f"(length {columns.shape[1]}) to {filepath}"
    )  # Confirm the export.


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    image_path = (
        r"C:\Users\Alix\Desktop\STG2\documents for datasets"
        r"\B-scans\B-scans\data_1.png"
    )  # Define the path to the B-scan.

    bscan_gray = load_bscan_grayscale(
        image_path
    )  # Load the B-scan.

    print(
        f"Loaded image: shape = "
        f"{bscan_gray.shape} (rows x cols)"
    )  # Display the image dimensions.

    columns = extract_columns(
        bscan_gray,
        normalize=True,
        assume_diverging_grey=True
    )  # Extract one A-scan from each B-scan column.

    print(
        f"Extracted {columns.shape[0]} columns "
        f"(A-scans) of length {columns.shape[1]}"
    )  # Display the number of extracted A-scans.

    H, W = bscan_gray.shape  # Get the B-scan dimensions.

    time_axis_us = None  # No physical time axis is defined yet.
    position_axis_mm = None  # No physical position axis is defined yet.

    # time_axis_us = build_time_axis(
    #     H,
    #     time_min_us=45.0,
    #     time_max_us=60.0
    # )  # Uncomment this if the real time range is known.

    # position_axis_mm = build_position_axis(
    #     W,
    #     pos_min_mm=0.0,
    #     pos_max_mm=300.0
    # )  # Uncomment this if the real scanning range is known.

    save_columns_as_ascan_dataset(
        columns,
        "extracted_ascans_from_bscan.npy"
    )  # Save all A-scans.

    interactive_ascan_viewer(
        bscan_gray=bscan_gray,
        columns=columns,
        depth_axis=time_axis_us,
        position_axis=position_axis_mm,
        history_length=10
    )  # Start the interactive viewer.

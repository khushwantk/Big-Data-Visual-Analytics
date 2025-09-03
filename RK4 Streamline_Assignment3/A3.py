import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def runge_kutta_integration(vector_func, start_point, step, steps, bounds):
    path_points = [start_point]
    current_point = start_point
    for i in range(steps):
        k1 = step * vector_func(current_point)
        k2 = step * vector_func(current_point + k1 / 2)
        k3 = step * vector_func(current_point + k2 / 2)
        k4 = step * vector_func(current_point + k3)
        next_point = current_point + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Check whether the next point is within the data bounds.
        if not (bounds[0] <= next_point[0] <= bounds[1] and
                bounds[2] <= next_point[1] <= bounds[3] and
                bounds[4] <= next_point[2] <= bounds[5]):
            print(f"Stopped integration at step {i} as point {next_point} is outside bounds.")
            break

        path_points.append(next_point)
        current_point = next_point

    return path_points

def sample_vector_at_point(vtk_data, sample_point):
    # Extract the vector value at a given point in the vector field using vtkProbeFilter
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetSourceData(vtk_data)

    sample_points = vtk.vtkPoints()
    sample_points.InsertNextPoint(sample_point.tolist())
    sample_polydata = vtk.vtkPolyData()
    sample_polydata.SetPoints(sample_points)

    probe_filter.SetInputData(sample_polydata)
    probe_filter.Update()

    output_data = probe_filter.GetOutput()
    vectors_array = vtk_to_numpy(output_data.GetPointData().GetArray("vectors"))

    if vectors_array is None:
        return np.array([0.0, 0.0, 0.0])
    else:
        return vectors_array[0]

def load_vector_field(filename):
    # Load a VTK vector field dataset from a .vti file
    print("Loading vector field from:", filename)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    print("Vector field loaded successfully. Bounds:", data.GetBounds())
    return data

def compute_streamline(vector_field, seed_point, step_size, max_steps):
    # Compute the streamline in both forward and backward directions from the seed point
    bounds = vector_field.GetBounds()
    print("Computing forward streamline...")
    forward_streamline = runge_kutta_integration(
        lambda p: sample_vector_at_point(vector_field, p),
        seed_point, step_size, max_steps, bounds
    )

    print("Computing backward streamline...")
    backward_streamline = runge_kutta_integration(
        lambda p: sample_vector_at_point(vector_field, p),
        seed_point, -step_size, max_steps, bounds
    )

    # Reverse the backward streamline and combine with the forward one.
    backward_array = np.array(backward_streamline)
    forward_array = np.array(forward_streamline)
    combined = np.concatenate((backward_array[::-1], forward_array))
    print("Streamline generated with total", combined.shape[0], "points.")
    return combined

def save_streamline_to_vtp(points_array, output_filename):
    # Save the computed streamline points to a .vtp file
    print("Saving streamline to file:", output_filename)
    vtk_points = vtk.vtkPoints()
    vtk_lines = vtk.vtkCellArray()

    for idx, pt in enumerate(points_array):
        vtk_points.InsertNextPoint(pt.tolist())
        if idx > 0:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, idx - 1)
            line.GetPointIds().SetId(1, idx)
            vtk_lines.InsertNextCell(line)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(poly_data)
    writer.Write()
    print("Streamline saved successfully.")
    return poly_data

def parse_arguments():
    # Parse the command-line arguments to obtain the seed point coordinates.
    if len(sys.argv) not in [4, 5]:
        print("Usage: python <script_name.py> <x> <y> <z> [yes/no]")
        sys.exit(1)

    try:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
        z = float(sys.argv[3])
    except ValueError:
        print("Error: Seed point values must be valid numbers.")
        sys.exit(1)

    viz = str(sys.argv[4]) if len(sys.argv) == 5 else "no"
    seed = np.array([x, y, z])
    print(f"Seed point: ({x}, {y}, {z})")
    return seed,viz


def display_polydata(polydata_object):
    # Read the VTP file containing the streamline
    data_mapper = vtk.vtkPolyDataMapper()
    data_mapper.SetInputData(polydata_object)

    # Create an actor to represent the polydata.
    data_actor = vtk.vtkActor()
    data_actor.SetMapper(data_mapper)
    data_actor.GetProperty().SetLineWidth(2)  # Increase line width for better visibility
    data_actor.GetProperty().SetColor(0, 1, 0)  # Set actor color to green (R, G, B)

    # Create a renderer and add the actor to it.
    renderer = vtk.vtkRenderer()
    renderer.AddActor(data_actor)
    renderer.SetBackground(1, 1, 1)  # Set a black background

    # Create a render window, assign the renderer, and set the window size.
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 1000)

    # Create a render window interactor for handling user inputs.
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Render the scene and start the interaction loop.
    render_window.Render()
    render_window_interactor.Start()


def main():
    # Get seed point from command-line arguments
    seed_point,viz = parse_arguments()
    print("===== Streamline Generation Started =====")

    # Set the file name for the input vector field
    vector_field_file = "tornado3d_vector.vti"
    vector_field = load_vector_field(vector_field_file)



    # Set integration parameters
    step_size = 0.05
    max_steps = 1000

    # Compute the streamline (both forward and backward)
    print("-"*50)
    streamline_points = compute_streamline(vector_field, seed_point, step_size, max_steps)
    print("-"*50)

    # Save the computed streamline to a .vtp file
    output_file = "streamline.vtp"
    polydata=save_streamline_to_vtp(streamline_points, output_file)

    print("===== Streamline Generation Completed =====")


    print("-"*50)


    if(viz=="yes"):
        print("Vizualizing...")
        display_polydata(polydata)

if __name__ == "__main__":
    main()

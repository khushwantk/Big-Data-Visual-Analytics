# CS661 Assignment1 Part1- 2D Isocontour Extraction
# Constraints:
# - not allowed to use VTK’s isocontour filter or any filter from VTK
# - do not have to handle the cells that have ambiguities by the Asymptotic Decider
# - traverse the vertices of each cell in the counterclockwise order
# - Works for any isovalue in (-1438, 630)

import vtk
import sys

# Read the VTK image data from file
def read_vtk_image(input_file):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_file)
    reader.Update()
    return reader.GetOutput()

# Retrieve scalar values of corners
def get_cell_values(image_data, i, j,dimension):
    scalars = image_data.GetPointData().GetScalars()
    # 1.Bottom-left, 2.Bottom-right, 3.Top-right,4.Top-left
    root=i + j * dimension[0]
    return [
        scalars.GetTuple1(root),
        scalars.GetTuple1(root+1),
        scalars.GetTuple1(root+dimension[0]+1),
        scalars.GetTuple1(root+dimension[0])
    ]

# Finding intersection point of the isovalue with cell edges
def find_intersection_points(corner_cordinates, corner_values, isovalue):
    pz = 0  # Constant z-coordinate [Can change to 25 to get exact 3D overlap with Paraview Iscontour of Dataset]
    intersection_points = []
    for k in range(4):
        # Edge 1-2,2-3,3-4,4-1 (As the cordinates are already passed in anticlock order)
        v1, v2 = corner_values[k], corner_values[(k+1) % 4]
        #(v1 < isovalue and v2 > isovalue) or (v1 > isovalue and v2 < isovalue)
        if (v1 - isovalue) * (v2 - isovalue) < 0:
            t = (isovalue - v1) / (v2 - v1)
            x1, y1 = corner_cordinates[k]
            x2, y2 = corner_cordinates[(k+1) % 4]
            intersection_points.append([x1 + t * (x2 - x1), y1 + t * (y2 - y1), pz])
    return intersection_points



# Generate the isocontour polydata based on the given isovalue
def generate_isocontour(image_data, isovalue):
    dimension = image_data.GetDimensions()

    # Store intersection points and isocontour lines
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    for i in range(dimension[0] - 1): #column (x)
        for j in range(dimension[1] - 1): #row (y)
            # Anticlockwise Traverse (i,j)(i+1,j)(i+1,j+1)(i, j+1)
            corner_cordinates = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
            corner_values = get_cell_values(image_data, i, j,dimension)

            intersection_points = find_intersection_points(corner_cordinates, corner_values, isovalue)

            # Create line segments for exactly two intersections (non-ambiguous)
            if len(intersection_points) == 2:
                pid1, pid2 = [points.InsertNextPoint(p) for p in intersection_points]
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, pid1)
                line.GetPointIds().SetId(1, pid2)
                lines.InsertNextCell(line)

            # Case for 4 intersections : FORMING LINES IN ANTI-CLOCK WISE DIRECTION
            # We already read the points already in anticlockwise manner so the lines will be in same order.
            elif len(intersection_points) == 4:
                pid1, pid2, pid3, pid4 = [points.InsertNextPoint(p) for p in intersection_points]
                for p1, p2 in [(pid1, pid2), (pid3, pid4)]:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, p1)
                    line.GetPointIds().SetId(1, p2)
                    lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    return polydata

# Save polydata to .vtp file
def save_polydata(polydata, isocontour_file):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(isocontour_file)
    writer.SetInputData(polydata)
    writer.Write()


# Visualize the isocontour from the output file
def visualize_contour(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 0)  # RGB
    actor.GetProperty().SetLineWidth(1)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 1000)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    interactor.Start()

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 file.py <input_data_file> <isovalue> <output.vtp>")
        sys.exit(1)

    input_file = sys.argv[1]
    isovalue = float(sys.argv[2])
    isocontour_file = str(sys.argv[3])

#  Acc to Paraview Value Ranges are from (-1434.86,630.569)
    if not -1438 <= isovalue <= 630:
        print("Error: Isovalue must be within (-1438, 630)")
        exit(1)

    print(f"Input file path: {input_file}")
    print(f"Isovalue: {isovalue}")

    image_data = read_vtk_image(input_file)
    polydata = generate_isocontour(image_data, isovalue)
    save_polydata(polydata, isocontour_file)

    print(f"Isocontour saved as: {isocontour_file}")
    visualize_contour(isocontour_file)

if __name__ == "__main__":
    main()

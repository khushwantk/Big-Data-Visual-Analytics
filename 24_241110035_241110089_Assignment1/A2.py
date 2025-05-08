# CS661 Assignment1 Part2 - Volume Rendering with VTK
import vtk
import sys


def parse_arguments():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <input_file.vti> [phong (0 or 1), default=0]")
        sys.exit(1)
    input_file = sys.argv[1]
    phong = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    return input_file, phong


def load_vti_file(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader


def create_transfer_functions():
    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(-4931.54, 0.0, 1.0, 1.0)
    color_tf.AddRGBPoint(-2508.95, 0.0, 0.0, 1.0)
    color_tf.AddRGBPoint(-1873.9, 0.0, 0.0, 0.5)
    color_tf.AddRGBPoint(-1027.16, 1.0, 0.0, 0.0)
    color_tf.AddRGBPoint(-298.031, 1.0, 0.4, 0.0)
    color_tf.AddRGBPoint(2594.97, 1.0, 1.0, 0.0)

    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(-4931.54, 1.0)
    opacity_tf.AddPoint(101.815, 0.002)
    opacity_tf.AddPoint(2594.97, 0.0)

    return color_tf, opacity_tf


def configure_volume_properties(phong_enabled):
    color_tf, opacity_tf = create_transfer_functions()
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_tf)
    volume_property.SetScalarOpacity(opacity_tf)
    volume_property.SetInterpolationTypeToLinear()

    if phong_enabled:
        volume_property.ShadeOn()
        volume_property.SetAmbient(0.5)
        volume_property.SetDiffuse(0.5)
        volume_property.SetSpecular(0.5)

    return volume_property

def create_volume(reader, volume_property):
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)

    return volume

def create_outline(reader):
    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(reader.GetOutputPort())

    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline.GetOutputPort())

    outline_actor = vtk.vtkActor()
    outline_actor.SetMapper(outline_mapper)
    # Black outline to the Volume Render
    outline_actor.GetProperty().SetColor(0, 0, 0)

    return outline_actor

def setup_renderer(volume, outline_actor):
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.AddActor(outline_actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1000, 1000)
    render_window.AddRenderer(renderer)

    return render_window, renderer

def setup_interactor(render_window):
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    return interactor

def save_screenshot(render_window, renderer, filename):
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetScale(1)
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

def capture_views(render_window, renderer):
    renderer.GetActiveCamera().Azimuth(0)
    render_window.Render()
    save_screenshot(render_window, renderer, "back_view.png")

    renderer.GetActiveCamera().Azimuth(180)
    render_window.Render()
    save_screenshot(render_window, renderer, "front_view.png")

def main():
    input_file, phong = parse_arguments()
    reader = load_vti_file(input_file)
    volume_property = configure_volume_properties(phong)
    volume = create_volume(reader, volume_property)
    outline_actor = create_outline(reader)
    render_window, renderer = setup_renderer(volume, outline_actor)
    interactor = setup_interactor(render_window)

    render_window.Render()
    capture_views(render_window, renderer)

    interactor.Initialize()
    interactor.Start()

if __name__ == "__main__":
    main()

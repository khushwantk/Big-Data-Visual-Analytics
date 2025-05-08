# CS661 Assignment 1 Group 24
24_241110035_241110089_Assignment1
Khushwant Kaswan - 241110035 - khushwantk24@iitk.ac.in

Senthil Ganesh - 241110089 - senthil24@iitk.ac.in

---

1. Install VTK:
   ```bash
   pip install vtk
   ```
2. Datasets:
   - `Isabel_2D.vti` and `Isabel_3D.vti` should be in "Data" folder in the working directory or provide full absolute path as input.

## **Part 1: 2D Isocontour Extraction**
### Script: `A1.py`

#### **Usage**
```bash
python3 A1.py <input_data_file_path> <isovalue> <output_file_name.vtp>
```


#### **Parameters**
- `input_data_file_path`: Path to `Isabel_2D.vti` file
- `isovalue`: Scalar value in (-1438, 630).
- `output_file_name.vtp`: Name to save the output `.vtp` file in the same folder.

#### **Example**
```bash
python3 A1.py Data/Isabel_2D.vti 100 output.vtp
```
The script will open an interactive window automatically vizualizing the extracted isocontour.


---

## **Part 2: Volume Rendering**
### Script: `A2.py`

#### **Usage**
```bash
python3 A2.py <input_data_file_path> <phong (0 or 1)>
```

#### **Parameters**
- `input_file`: Path to `Isabel_3D.vti` file
- `phong`:
  - `1`: Enable
  - `0`: Disable (default)

#### **Example**
```bash
python3 A2.py Data/Isabel_3D.vti
python3 A2.py Data/Isabel_3D.vti 0
python3 A2.py Data/Isabel_3D.vti 1
```
The script will open an interactive window automatically and save 2 files "front_view.png" and "back_view.png".

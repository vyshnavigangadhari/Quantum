# Quantum TSQS Project – Setup and Execution Guide

## 1. Project Overview

This project implements a **Hybrid-Enhanced Two-Step Quantum Search (TSQS)** approach for solving the **Traveling Salesman Problem (TSP)** using the **Qiskit quantum computing framework**.

The implementation combines:

* Classical preprocessing techniques (nearest neighbor, 2-opt optimization)
* Quantum amplitude amplification
* Threshold-based oracle filtering
* Adaptive iteration tuning
* Quantum simulation using **Qiskit Aer**

The program generates candidate tours, applies quantum search amplification, and visualizes the probability distribution of the resulting tours.

---

# 2. System Requirements

The following software environment was used to run the project successfully.

| Component           | Version                              |
| ------------------- | ------------------------------------ |
| Python              | 3.11                                 |
| Package Manager     | pip                                  |
| Environment Manager | venv (Python Virtual Environment)    |
| Quantum Framework   | Qiskit                               |
| Simulator Backend   | Qiskit Aer                           |
| Numerical Library   | NumPy                                |
| Visualization       | Matplotlib                           |
| System Dependency   | Microsoft Visual C++ Redistributable |

---

# 3. Python Installation

Python **3.11** was used because the `qiskit-aer` simulator depends on compiled binary components that are not fully supported in newer Python versions (such as Python 3.13 or 3.14).

Python was downloaded from:

https://www.python.org/downloads/

During installation, the option **“Add Python to PATH”** was enabled.

---

# 4. Creating a Virtual Environment

A **virtual environment** was created to isolate the project dependencies and avoid conflicts with globally installed Python packages.

Create the virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows (PowerShell)

```bash
.venv\Scripts\activate
```

Once activated, the terminal prompt will appear like:

```
(.venv) PS C:\ProjectFolder>
```

All packages are installed inside this environment.

---

# 5. Installing Required Dependencies

The required Python libraries were installed using `pip`.

```bash
pip install qiskit
pip install qiskit-aer
pip install numpy
pip install matplotlib
```

### Package Roles

| Package    | Purpose                            |
| ---------- | ---------------------------------- |
| qiskit     | Core quantum computing framework   |
| qiskit-aer | High-performance quantum simulator |
| numpy      | Numerical operations               |
| matplotlib | Visualization of results           |

---

# 6. Microsoft Visual C++ Redistributable Requirement

The **Qiskit Aer simulator** contains compiled C++ extensions.
Windows requires the **Microsoft Visual C++ runtime libraries** to load these compiled modules.

Without this dependency, the following error may occur:

```
ImportError: DLL load failed while importing controller_wrappers
```

To resolve this issue, the **Microsoft Visual C++ Redistributable (x64)** was installed.

Download from:

https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

Install:

```
VC_redist.x64.exe
```

After installation, restart the system if required.

---

# 7. Running the Program

After installing all dependencies and activating the virtual environment, run the program using:

```bash
python tsqs_increment2.py
```

The program performs the following steps:

1. Preprocesses TSP tours using classical heuristics
2. Builds a quantum circuit implementing the TSQS algorithm
3. Simulates the circuit using the Qiskit Aer simulator
4. Measures and decodes results
5. Displays probability distributions of possible tours
6. Visualizes results using matplotlib

---

# 8. Output

The program outputs:

* Tour summaries with cost and probability
* Optimal tour detection
* Runtime statistics
* Circuit metrics

Example output:

```
=== Tours Summary (Hybrid TSQS) ===
tour=(1,2,3) cost=9 freq=1200 p=0.586 <-- OPTIMAL
tour=(1,3,2) cost=11 freq=848 p=0.414
```

The program also generates visualizations showing:

* Tour frequency distribution
* Tour probability distribution

---

# 9. Why Virtual Environment Was Used

A **Python virtual environment (venv)** was used instead of Conda because:

* Lightweight and built into Python
* Direct integration with `pip`
* Easy interpreter selection in VS Code
* Avoids installing additional environment managers

---

# 10. Final Working Configuration

The project was successfully executed with the following setup:

| Component      | Configuration                        |
| -------------- | ------------------------------------ |
| Python Version | 3.11                                 |
| Environment    | Python venv                          |
| Dependencies   | Installed via pip                    |
| Simulator      | Qiskit Aer                           |
| System Library | Microsoft Visual C++ Redistributable |

With this configuration, the **DLL import errors were resolved** and the TSQS quantum simulation ran successfully.

---

# 11. Notes

* Using **Python versions newer than 3.11** may cause compatibility issues with `qiskit-aer`.
* Always install dependencies inside the **virtual environment**.
* Ensure **Microsoft Visual C++ Redistributable** is installed for Windows systems.

---

# 12. Author

Project implemented as part of a **Quantum Computing experiment on hybrid classical-quantum optimization for the Traveling Salesman Problem (TSP)**.

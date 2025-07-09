# Engineering Standards

## Important IP and Legal Notice

**This repository contains code that implements formulas and calculations derived from engineering standards, but does not reproduce, copy, or distribute the actual content of those standards.** The functions and implementations in this library represent our interpretation and coding of the mathematical formulas and data tables found in various Australian Standards (AS), New Zealand Standards (NZS), and other engineering standards.

**Key Points:**
- This code represents formulas and data embodied in the standards, not the standards themselves
- No copyrighted content from the standards documents is reproduced here
- Users are responsible for obtaining proper licenses for the actual standards documents
- This library is provided as a tool to assist with calculations based on publicly available formulas
- The standards organizations retain all rights to their published documents

## About

This Python library provides function implementations for calculations based on various engineering standards, primarily focused on Australian and New Zealand structural engineering standards. The library is organized into two main packages:

- `as_standards`: Functions based on Australian Standards
- `nz_standards`: Functions based on New Zealand Standards

## Installation and Usage

### Prerequisites

This library requires Python 3.6 or higher and depends on:
- NumPy
- Pandas

### Installation Options

#### Option 1: Install from Source (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/engineering-standards.git
   cd engineering-standards
   ```

2. Install the package:
   ```bash
   pip install .
   ```

3. Or install in development mode if you plan to contribute:
   ```bash
   pip install -e .
   ```

#### Option 2: Install Dependencies Manually

If you prefer to use the library without installing it as a package:

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install numpy pandas
   ```

3. Add the repository to your Python path or copy the modules to your project

### Usage Examples

#### Basic Usage

```python
# Import specific standards modules
from as_standards import AS_1170_2_2021
from nz_standards import NZS_1170_5_2004

# Use functions from the imported modules
# Example: Calculate wind pressure using AS 1170.2
pressure = AS_1170_2_2021.calculate_wind_pressure(velocity=45, height=10)
```

#### Common Development Environments

**Jupyter Notebooks:**
```python
import sys
sys.path.append('/path/to/engineering-standards')

from as_standards import AS_1170_2_2021
# Your calculations here
```

**Python Scripts:**
```python
#!/usr/bin/env python3
from engineering_standards.as_standards import AS_1170_2_2021
from engineering_standards.nz_standards import NZS_1170_5_2004

# Your engineering calculations
```

**Virtual Environment (Recommended):**
```bash
python -m venv engineering_env
source engineering_env/bin/activate  # On Windows: engineering_env\Scripts\activate
pip install -e /path/to/engineering-standards
```

## Available Standards

### Australian Standards (as_standards)
- AS 1170.2 (Wind Actions)
- AS 1170.4 (Earthquake Actions)
- AS 4084 (Steel Storage Racking)
- AS/NZS 1170.0 (Structural Design Actions)
- AS/NZS 1170.2 (Wind Actions)
- AS/NZS 1664.1 (Aluminum Structures)
- AS/NZS 4600 (Cold-formed Steel)
- AS/NZS 1720.1 (Timber Structures)

### New Zealand Standards (nz_standards)
- NZS 1170.2 (Wind Actions)
- NZS 1170.5 (Earthquake Actions)
- NZS 3101.1 (Concrete Structures)
- AS/NZS 1170.0 (Structural Design Actions)
- AS/NZS 4600 (Cold-formed Steel)
- Building Code Clauses F4 & F9

## License

This project is licensed under the MIT License, which is a permissive open-source license.

### What does this mean?

The MIT License is one of the most permissive and widely-used open-source licenses. Here's what it means for users:

**You CAN:**
- Use this software for any purpose (commercial or non-commercial)
- Modify the code to suit your needs
- Distribute the original or modified code
- Include this code in proprietary software
- Sell products that include this code

**You MUST:**
- Include the original copyright notice and license text in any substantial portions of the software you distribute
- Provide attribution to the original authors

**You CANNOT:**
- Hold the authors liable for any damages or issues arising from the use of this software
- Expect any warranty or guarantee about the software's performance

**In Simple Terms:**
This is free software that you can use however you want, but the authors aren't responsible if something goes wrong, and you need to give credit where it's due.

The full license text is available in the [LICENSE](LICENSE) file.

## Contributing

We welcome contributions to improve and extend this library! Here's how you can contribute:

### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/engineering-standards.git
   cd engineering-standards
   ```

3. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. Install in development mode:
   ```bash
   pip install -e .
   ```

### Types of Contributions

**Code Contributions:**
- Implement new standards or extend existing ones
- Fix bugs or improve existing implementations
- Add test cases and documentation
- Improve performance or code quality

**Documentation:**
- Improve README, docstrings, or inline comments
- Add usage examples or tutorials
- Create or update API documentation

**Testing:**
- Add test cases for existing functions
- Improve test coverage
- Report bugs or issues

### Contribution Guidelines

1. **Code Style:**
   - Follow PEP 8 Python style guidelines
   - Use descriptive variable and function names
   - Add docstrings to all functions and classes
   - Include type hints where appropriate

2. **Testing:**
   - Test your code thoroughly before submitting
   - Include test cases for new functions
   - Ensure existing tests still pass

3. **Documentation:**
   - Update docstrings for any modified functions
   - Add comments for complex calculations
   - Update this README if adding new standards

4. **Standards Implementation:**
   - Clearly reference the specific standard and section being implemented
   - Include the relevant equation numbers or table references in comments
   - Ensure calculations match the standard exactly

### Submitting Changes

1. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add AS 1170.3 snow load calculations"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request on GitHub with:
   - Clear description of changes
   - Reference to relevant standards or issues
   - Any testing performed

### Code of Conduct

- Be respectful and professional in all interactions
- Focus on constructive feedback and collaboration
- Respect the IP considerations mentioned in this README
- Ensure accuracy in engineering calculations

### Questions or Issues?

- Open an issue on GitHub for bugs or feature requests
- Use discussions for questions about usage or implementation
- Contact the maintainers for sensitive matters

## Disclaimer

This software is provided for educational and professional use. Users are responsible for verifying the accuracy of calculations and ensuring compliance with applicable standards and regulations. Always consult the original standards documents and qualified professionals for critical engineering decisions.

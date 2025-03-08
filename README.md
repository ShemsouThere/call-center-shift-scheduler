# Call Center Shift Scheduler

## Overview
This project was originally developed as an automated employee shift planning system for a Djezzy call center. While I've since moved on to other projects, this prototype is functional and demonstrates the use of constraint programming to create optimal shift schedules while respecting various business rules and employee constraints.

## Features
- Automated scheduling of employees across multiple shifts and days
- Respect for employee availability preferences
- Integration with employee database
- Enforcement of business rules:
  - Minimum and maximum staffing requirements per shift
  - Rest day requirements after night/late shifts
  - Single shift per day rule
  - Male worker assignment to night shifts
  - Balanced workload distribution (3 days per week)
- CSV output of generated schedules

## How It Works
The scheduler uses Google's OR-Tools constraint programming solver to find an optimal schedule that satisfies all hard constraints while minimizing violations of soft constraints. The system:

1. Loads employee data from a SQLite database
2. Retrieves employee availability preferences
3. Applies mandatory business rules as hard constraints
4. Models staffing preferences as soft constraints with penalties
5. Solves the constraint satisfaction problem
6. Outputs the schedule to a CSV file

## Technologies Used
- Python
- OR-Tools (Google's constraint programming library)
- SQLite
- Pandas
- CSV for input/output

## Required Files
The system requires the following files to operate:

1. `call_center.db` - SQLite database containing:
   - `employees` table with fields: Employee_Id, Status, First_Name, Last_Name, transport
   - `availability` table with fields: Employee_Id, Shift_1, Shift_2, Shift_3, Shift_4, Hours

2. `staffmin.csv` - CSV file defining shift constraints with fields:
   - Day (Dimanche, Lundi, Mardi, Mercredi, Jeudi, Vendredi, Samedi)
   - Shift_1_Min, Shift_1_Max
   - Shift_2_Min, Shift_2_Max
   - Shift_3_Min, Shift_3_Max
   - Shift_4_Min, Shift_4_Max
   - Total_Min, Total_Max

3. `manual_shifts.csv` - CSV file containing manually assigned shifts

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/call-center-scheduler.git
cd call-center-scheduler

# Install dependencies
pip install ortools pandas
```

## Usage

```bash
# Run the scheduler
python scheduler.py
```

The program will output the generated schedule to `shift_schedule_and_counts.csv`.

## Database Schema

### Employees Table
| Field | Type | Description |
|-------|------|-------------|
| Employee_Id | INTEGER | Unique identifier for each employee |
| Status | TEXT | Employment status (Full-time, Part-time, etc.) |
| First_Name | TEXT | Employee's first name |
| Last_Name | TEXT | Employee's last name |
| transport | TEXT | Transportation method used by employee |

### Availability Table
| Field | Type | Description |
|-------|------|-------------|
| Employee_Id | INTEGER | Employee identifier (foreign key) |
| Shift_1 | INTEGER | Availability for morning shift (0=unavailable, 1=available) |
| Shift_2 | INTEGER | Availability for afternoon shift |
| Shift_3 | INTEGER | Availability for evening shift |
| Shift_4 | INTEGER | Availability for night shift |
| Hours | INTEGER | Maximum hours available to work |

## Shift Definitions
The system defines 4 standard shifts:
- Shift 1: 08:00-16:30
- Shift 2: 12:00-20:30
- Shift 3: 16:30-00:00
- Shift 4: 00:00-08:00 (Night shift)

## Business Rules
1. All employees work exactly 3 days per week
2. After a late shift (Shift 3), employees must have a rest day
3. After a night shift (Shift 4), employees must have a rest day
4. Only male employees are assigned to night shifts
5. Only one night shift allowed per week per employee
6. Employees may only work one shift per day

## Future Improvements
- Add a web interface for easier interaction
- Implement employee preferences beyond basic availability
- Add support for part-time workers with different weekly hour requirements
- Enhance optimization to balance workload more evenly
- Add support for holidays and time-off requests
- Improve error handling and validation

## Contact
shemsoudev@gmail.com

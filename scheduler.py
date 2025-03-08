from ortools.sat.python import cp_model
import csv, sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random


class CallCenterShiftScheduler:
    def __init__(self):
        # Connect to the database
        self.conn = sqlite3.connect('call_center.db')
        self.cursor = self.conn.cursor()
   
        # Load employees from the database
        self.employees_df = pd.read_sql_query('SELECT Employee_Id, Status, First_Name, Last_Name, transport FROM employees', self.conn)
        self.cursor.execute('SELECT Employee_Id, Status, First_Name, Last_Name, transport FROM employees')
        self.employees = self.cursor.fetchall()
        self.num_workers = len(self.employees)

        # Load availability from the database
        self.cursor.execute('SELECT * FROM availability')
        self.availability = self.cursor.fetchall()
        
        # Load manual shifts from CSV
        self.manual_shifts_df = pd.read_csv('manual_shifts.csv')

        # Convert availability to a DataFrame for easier access
        self.availability_df = pd.DataFrame(self.availability, columns=['Employee_Id', 'Shift_1', 'Shift_2', 'Shift_3', 'Shift_4', 'Hours'])

        # Default shift times
        self.num_shifts = 4
        self.num_days = 7
        self.shift_times = {
            0: '08:00-16:30',
            1: '12:00-20:30',
            2: '16:30-00:00',
            3: '00:00-08:00'
        }

        # If num_shifts is less than default, slice the dictionary
        if self.num_shifts < len(self.shift_times):
            self.shift_times = {k: v for k, v in self.shift_times.items() if k < self.num_shifts}

        # Initialize gender information (assuming all employees are male)
        self.is_male = [True] * self.num_workers

        # Load shift constraints
        self.shift_constraints = self.load_shift_constraints('staffmin.csv')

        # Max iterations for solver
        self.max_iterations = 1000
        
        # Pre-check feasibility of the problem
        self.check_basic_feasibility()

    def check_basic_feasibility(self):
        """Check if basic staffing requirements can be met given worker availability"""
        print("Checking basic feasibility of scheduling problem...")
        
        # Count total available shifts per employee
        available_shifts_per_worker = []
        for i in range(self.num_workers):
            employee_id = self.employees[i][0]
            # Fetch availability for the employee
            self.cursor.execute('SELECT Shift_1, Shift_2, Shift_3, Shift_4 FROM availability WHERE Employee_Id = ?', (employee_id,))
            availability = self.cursor.fetchone()
            if availability:
                total_available = sum(availability[j] for j in range(self.num_shifts)) * self.num_days
                available_shifts_per_worker.append(total_available)
            else:
                available_shifts_per_worker.append(0)
        
        total_available_shifts = sum(available_shifts_per_worker)
        print(f"Total available worker-shifts: {total_available_shifts}")
        
        # Calculate minimum required shifts based on constraints
        day_names = ['Dimanche', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi']
        min_required_shifts = 0
        for k in range(self.num_days):
            day_name = day_names[k]
            constraints = self.shift_constraints.get(day_name, {})
            min_required_shifts += constraints.get('Total_Min', 0)
        
        print(f"Minimum required shifts: {min_required_shifts}")
        
        # Check if we have enough worker capacity
        if total_available_shifts < min_required_shifts:
            print("WARNING: Not enough worker availability to meet minimum requirements!")
            print("Consider adjusting constraints or increasing worker availability.")
        else:
            min_shifts_per_worker = 3  # Each worker should have at least 3 shifts
            max_workers_capacity = min(self.num_workers * min_shifts_per_worker, total_available_shifts)
            if max_workers_capacity < min_required_shifts:
                print("WARNING: Worker capacity may be insufficient with 3-day limit!")
                print(f"Maximum capacity with 3 shifts per worker: {max_workers_capacity}")
                print(f"Consider allowing more shifts per worker or adding more workers.")

    def load_shift_constraints(self, filename):
        constraints = {}
        required_keys = ['Day', 'Shift_1_Min', 'Shift_1_Max', 'Shift_2_Min', 'Shift_2_Max',
                         'Shift_3_Min', 'Shift_3_Max', 'Shift_4_Min', 'Shift_4_Max',
                         'Total_Min', 'Total_Max']
        try:
            with open(filename, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if not all(key in row for key in required_keys):
                        print(f"Missing keys in row: {row}")
                        continue
                    day = row['Day']
                    shift_min_max = {}
                    for i in range(1, self.num_shifts + 1):
                        shift_min_key = f'Shift_{i}_Min'
                        shift_max_key = f'Shift_{i}_Max'
                        shift_min_max[f'Shift_{i}_Min'] = int(row[shift_min_key])
                        shift_min_max[f'Shift_{i}_Max'] = int(row[shift_max_key])
                    shift_min_max['Total_Min'] = int(row['Total_Min'])
                    shift_min_max['Total_Max'] = int(row['Total_Max'])
                    constraints[day] = shift_min_max
        except Exception as e:
            print(f"Error reading shift constraints CSV: {e}")
            exit(1)
        return constraints
    
    def apply_availability_constraints(self, model, shifts):
        # Define the order of days corresponding to k in shifts[i][j][k]
        days_order = ['Dimanche', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi']
        
        for i in range(self.num_workers):
            employee_id = self.employees[i][0]
            # Fetch availability for the employee
            self.cursor.execute('SELECT Shift_1, Shift_2, Shift_3, Shift_4 FROM availability WHERE Employee_Id = ?', (employee_id,))
            availability = self.cursor.fetchone()
            if availability:
                for k in range(self.num_days):
                    for j in range(self.num_shifts):
                        if availability[j] == 0:
                            # Employee is unavailable for this shift on this day
                            model.Add(shifts[i][j][k] == 0)

    def create_model_with_relaxation(self, relaxation_level=0, allow_more_shifts=False):
        # Create a new model for each attempt
        model = cp_model.CpModel()
        shifts = [[[model.NewBoolVar(f'shift_{i}_{j}_{k}') for k in range(self.num_days)]
                  for j in range(self.num_shifts)] for i in range(self.num_workers)]
        
        # Apply strict availability constraints - these can't be compromised
        self.apply_availability_constraints(model, shifts)
        
        # Define the days of the week
        day_names = ['Dimanche', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi']
        
        # Create penalty variables for all soft constraints
        # 1. Penalty for exceeding max workers per shift
        excess_workers = [[model.NewIntVar(0, self.num_workers, f'excess_workers_{j}_{k}')
                          for k in range(self.num_days)] for j in range(self.num_shifts)]
        
        # 2. Penalty for not meeting minimum workers per shift
        min_workers_deficit = [[model.NewIntVar(0, self.num_workers, f'min_workers_deficit_{j}_{k}')
                              for k in range(self.num_days)] for j in range(self.num_shifts)]
        
        # 3. Penalty for worker shift deviation
        worker_shift_deficit = [model.NewIntVar(0, 3, f'worker_shift_deficit_{i}') 
                               for i in range(self.num_workers)]
        worker_shift_excess = [model.NewIntVar(0, 7, f'worker_shift_excess_{i}') 
                              for i in range(self.num_workers)]
        
        # 4. Penalty for total day staffing
        day_total_deficit = [model.NewIntVar(0, self.num_workers * self.num_shifts, f'day_total_deficit_{k}')
                            for k in range(self.num_days)]
        day_total_excess = [model.NewIntVar(0, self.num_workers * self.num_shifts, f'day_total_excess_{k}')
                           for k in range(self.num_days)]
        
        # 5. Critical constraints with high penalties instead of hard constraints
        consecutive_night_penalty = [model.NewIntVar(0, 7, f'consecutive_night_penalty_{i}')
                                   for i in range(self.num_workers)]
        
        # Apply shift constraints with relaxation for each day and shift
        for k in range(self.num_days):
            day_name = day_names[k]
            constraints = self.shift_constraints.get(day_name, {})
            
            # Total shifts per day
            total_min = constraints.get('Total_Min', 0)
            total_max = constraints.get('Total_Max', self.num_workers * self.num_shifts)
            
            # Apply with relaxation
            adjusted_total_min = max(0, total_min - relaxation_level)
            
            # Total shifts for the day
            day_total = sum(shifts[i][j][k] for i in range(self.num_workers) for j in range(self.num_shifts))
            
            # Use deficit/excess variables instead of hard constraints
            model.Add(day_total + day_total_deficit[k] >= adjusted_total_min)
            model.Add(day_total <= total_max + day_total_excess[k])
            
            # Apply shift-specific constraints
            for j in range(self.num_shifts):
                min_required = constraints.get(f'Shift_{j+1}_Min', 0)
                max_allowed = constraints.get(f'Shift_{j+1}_Max', self.num_workers)
                
                # Apply relaxation to minimum requirements
                adjusted_min = max(0, min_required - relaxation_level)
                
                # Count workers in this shift
                shift_total = sum(shifts[i][j][k] for i in range(self.num_workers))
                
                # Use deficit/excess variables
                model.Add(shift_total + min_workers_deficit[j][k] >= adjusted_min)
                model.Add(shift_total <= max_allowed + excess_workers[j][k])
                
                # Special night shift rule (at least one male, if possible)
                if j == 3:  # Night shift
                    night_male_count = sum(shifts[i][j][k] for i in range(self.num_workers) if self.is_male[i])
                    # Try to have at least one male, but allow relaxation
                    night_male_deficit = model.NewIntVar(0, 1, f'night_male_deficit_{k}')
                    model.Add(night_male_count + night_male_deficit >= 1)
        
        # Worker assignment constraints
        for i in range(self.num_workers):
            # Count total shifts for this worker
            worker_total_shifts = sum(shifts[i][j][k] for j in range(self.num_shifts) for k in range(self.num_days))
            
            # Target of 3 shifts per worker with possible deviation
            if allow_more_shifts and relaxation_level >= 2:
                # In higher relaxation levels, allow 3-4 shifts
                model.Add(worker_total_shifts + worker_shift_deficit[i] >= 3)
                model.Add(worker_total_shifts <= 4 + worker_shift_excess[i])
            else:
                # Standard 3 shifts with deviation
                model.Add(worker_total_shifts + worker_shift_deficit[i] >= 3)
                model.Add(worker_total_shifts <= 3 + worker_shift_excess[i])
            
            # Single shift per day - critical health constraint
            for k in range(self.num_days):
                model.Add(sum(shifts[i][j][k] for j in range(self.num_shifts)) <= 1)
            
            # Rest after night shift (shift 3) - softer with penalty
            for k in range(self.num_days - 1):
                night_shift = shifts[i][3][k]
                next_day_shifts = sum(shifts[i][j][k+1] for j in range(self.num_shifts))
                
                # Create a violation variable instead of hard constraint
                violation = model.NewBoolVar(f'night_rest_violation_{i}_{k}')
                model.Add(night_shift + next_day_shifts <= 1 + violation)
                # Track violations
                model.Add(consecutive_night_penalty[i] >= violation)
            
            # Rest after late shift (Shift 2) - softer constraint
            for k in range(self.num_days - 1):
                late_shift = shifts[i][2][k]
                next_day_early_shift = shifts[i][0][k+1]  # Early morning after late night
                
                # Create a violation variable
                late_violation = model.NewBoolVar(f'late_rest_violation_{i}_{k}')
                model.Add(late_shift + next_day_early_shift <= 1 + late_violation)
        
        # Calculate working days distribution to balance weekend work
        weekend_shifts = [sum(shifts[i][j][k] for j in range(self.num_shifts) for k in [0, 6]) 
                         for i in range(self.num_workers)]
        
        # Objective function with weighted penalties
        model.Minimize(
            # Deficit of minimum workers (most important) - high penalty
            sum(200 * min_workers_deficit[j][k] for j in range(self.num_shifts) for k in range(self.num_days)) +
            
            # Deficit of worker shifts (important) - high penalty
            sum(150 * worker_shift_deficit[i] for i in range(self.num_workers)) +
            
            # Deficit of total day staffing - medium penalty
            sum(100 * day_total_deficit[k] for k in range(self.num_days)) +
            
            # Excess workers (somewhat important) - medium penalty
            sum(50 * excess_workers[j][k] for j in range(self.num_shifts) for k in range(self.num_days)) +
            
            # Excess shifts for workers - medium penalty
            sum(50 * worker_shift_excess[i] for i in range(self.num_workers)) +
            
            # Night shift violations - critical health concern
            sum(300 * consecutive_night_penalty[i] for i in range(self.num_workers)) +
            
            # Balance weekend shifts - low penalty
            sum(10 * abs(weekend_shifts[i] - 1) for i in range(self.num_workers))
        )
        
        return model, shifts

    def solve(self):
        best_solution = None
        best_status = None
        best_shifts = None
        
        # Progressive strategy with increasing relaxation
        strategies = [
            {"relaxation_level": 0, "allow_more_shifts": False, "time_limit": 60},
            {"relaxation_level": 1, "allow_more_shifts": False, "time_limit": 90},
            {"relaxation_level": 2, "allow_more_shifts": False, "time_limit": 120},
            {"relaxation_level": 2, "allow_more_shifts": True, "time_limit": 150},
            {"relaxation_level": 3, "allow_more_shifts": True, "time_limit": 180},
        ]
        
        for strategy_idx, strategy in enumerate(strategies):
            print(f"Strategy {strategy_idx+1}: Relaxation={strategy['relaxation_level']}, "
                  f"Allow more shifts={strategy['allow_more_shifts']}")
            
            model, shifts = self.create_model_with_relaxation(
                relaxation_level=strategy['relaxation_level'],
                allow_more_shifts=strategy['allow_more_shifts']
            )
            
            # Set solver parameters
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = strategy['time_limit']
            solver.parameters.num_search_workers = 8
            
            # Enable solution logging - helps see progress of finding solutions
            if strategy_idx == len(strategies) - 1:  # Only for last attempt
                solver.parameters.log_search_progress = True
            
            # Solve the model
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                best_status = status
                best_shifts = shifts
                best_solution = solver
                print(f"Solution found with strategy {strategy_idx+1}")
                
                # For early strategies, continue to try to find better solutions
                if strategy_idx < 2:
                    print("Found a solution, but continuing to search for better ones...")
                else:
                    break  # Stop once we find a solution in later strategies
        
        if best_solution:
            status_str = "Optimal" if best_status == cp_model.OPTIMAL else "Feasible"
            print(f"{status_str} solution found.")
            self.print_solution(best_solution, best_shifts)
            return True
        else:
            print("No solution found after all attempts. Generating an approximate schedule...")
            # Generate a smart feasible solution as fallback
            success = self.generate_smart_fallback_schedule()
            return success

    def print_solution(self, solver, shifts):
        # Calculate shift counts
        constraint_violations = []
        
        for i in range(self.num_workers):
            # Check for worker shift count constraint (each worker should have 3 shifts)
            worker_shifts = sum(solver.BooleanValue(shifts[i][j][k]) 
                               for j in range(self.num_shifts) for k in range(self.num_days))
            
            if worker_shifts != 3:
                constraint_violations.append(f"Worker {self.employees[i][0]} has {worker_shifts} shifts instead of 3")
            
            # Check for rest after night shift
            for k in range(self.num_days - 1):
                if solver.BooleanValue(shifts[i][3][k]) and any(solver.BooleanValue(shifts[i][j][k+1]) 
                                                             for j in range(self.num_shifts)):
                    constraint_violations.append(f"Worker {self.employees[i][0]} has no rest after night shift on day {k}")
        
        # Print constraint violations if any
        if constraint_violations:
            print("\nWARNING: Some constraints were relaxed in the solution:")
            for violation in constraint_violations[:10]:  # Limit to first 10
                print(f"- {violation}")
            if len(constraint_violations) > 10:
                print(f"...and {len(constraint_violations) - 10} more violations.")
        
        shift_counts = {j: [0] * self.num_days for j in range(self.num_shifts)}
        for k in range(self.num_days):
            for j in range(self.num_shifts):
                shift_counts[j][k] = sum(solver.BooleanValue(shifts[i][j][k]) for i in range(self.num_workers))
        
        # Check minimum staffing requirements
        day_names = ['Dimanche', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi']
        staffing_violations = []
        
        for k in range(self.num_days):
            day_name = day_names[k]
            constraints = self.shift_constraints.get(day_name, {})
            
            for j in range(self.num_shifts):
                min_required = constraints.get(f'Shift_{j+1}_Min', 0)
                if shift_counts[j][k] < min_required:
                    staffing_violations.append(
                        f"{day_name} Shift {j+1}: {shift_counts[j][k]} workers (minimum required: {min_required})")
        
        if staffing_violations:
            print("\nWARNING: Some minimum staffing requirements were not met:")
            for violation in staffing_violations:
                print(f"- {violation}")
        
        # Prepare data for CSV
        # Prepare individual worker schedules
        worker_schedules = []
        for i in range(self.num_workers):
            employee_id = self.employees_df.iloc[i]['Employee_Id']
            first_name = self.employees_df.iloc[i]['First_Name']
            last_name = self.employees_df.iloc[i]['Last_Name']
            worker_name = f"{first_name} {last_name}"
            schedule_row = {'Employee_Id': employee_id, 'WorkerName': worker_name}
            
            for k in range(self.num_days):
                assigned_shift = 'Off'
                for j in range(self.num_shifts):
                    if solver.BooleanValue(shifts[i][j][k]):
                        assigned_shift = f'{self.shift_times[j]}'
                        break
                schedule_row[day_names[k]] = assigned_shift
            
            worker_schedules.append(schedule_row)
        
        # Prepare shift counts data
        shift_counts_data = []
        for k in range(self.num_days):
            row = {'Day': day_names[k]}
            
            for j in range(self.num_shifts):
                count = shift_counts[j][k]
                min_required = self.shift_constraints.get(day_names[k], {}).get(f'Shift_{j+1}_Min', 0)
                
                # Highlight if below minimum
                label = f'Shift {j+1}'
                if count < min_required:
                    label += f' (MIN: {min_required})'
                
                row[label] = count
            
            shift_counts_data.append(row)
        
        # Export to CSV
        with open('shift_schedule_solution.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Employee_Id', 'WorkerName'] + day_names)
            writer.writeheader()
            writer.writerows(worker_schedules)
    
            file.write('\n\n')  # Blank line between sections
    
            # Dynamic fieldnames for shift counts including warning labels
            shift_count_fields = ['Day'] + list(shift_counts_data[0].keys())[1:]
            writer = csv.DictWriter(file, fieldnames=shift_count_fields)
            writer.writeheader()
            writer.writerows(shift_counts_data)
    
        print('Solution exported to shift_schedule_solution.csv')

    def check_assignment_feasibility(self, employee_id, shift, day, current_assignments):
        """Check if assigning an employee to a shift is feasible given current assignments"""
        # Check availability
        self.cursor.execute('SELECT Shift_1, Shift_2, Shift_3, Shift_4 FROM availability WHERE Employee_Id = ?', (employee_id,))
        availability = self.cursor.fetchone()
        
        if not availability or availability[shift] == 0:
            return False
        
        # Check if already assigned on this day
        if any(a['day'] == day and a['employee_id'] == employee_id for a in current_assignments):
            return False
        
        # Check rest after night shift
        if day > 0 and any(a['day'] == day-1 and a['shift'] == 3 and a['employee_id'] == employee_id 
                          for a in current_assignments):
            return False
        
        # Check no work day after late shift
        if day > 0 and any(a['day'] == day-1 and a['shift'] == 2 and a['employee_id'] == employee_id 
                          for a in current_assignments):
            return False
        
        # Check number of shifts already assigned
        shifts_count = sum(1 for a in current_assignments if a['employee_id'] == employee_id)
        if shifts_count >= 3:
            return False
        
        return True

    def generate_smart_fallback_schedule(self):
        """Generate a more sophisticated fallback schedule respecting key constraints"""
        print("Generating optimized fallback schedule...")
        
        day_names = ['Dimanche', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi']
        assignments = []
        
        # First pass: Prioritize critical shifts (night shifts and high-demand periods)
        # Process shifts in order of difficulty (night shift first, then others)
        shift_order = [3, 2, 1, 0]  # Night shift first, then evening, afternoon, morning
        
        for shift_idx, shift in enumerate(shift_order):
            for day in range(self.num_days):
                day_name = day_names[day]
                constraints = self.shift_constraints.get(day_name, {})
                min_required = constraints.get(f'Shift_{shift+1}_Min', 0)
                
                # Count current assignments for this shift/day
                current_count = sum(1 for a in assignments if a['day'] == day and a['shift'] == shift)
                
                # Need more assignments?
                needed = max(0, min_required - current_count)
                
                if needed > 0:
                    # Find candidates for this shift
                    candidates = []
                    for i in range(self.num_workers):
                        employee_id = self.employees[i][0]
                        if self.check_assignment_feasibility(employee_id, shift, day, assignments):
                            # For night shifts, prioritize male workers
                            priority = 1
                            if shift == 3 and self.is_male[i]:
                                priority = 0
                            candidates.append((priority, employee_id))
                    
                    # Sort candidates (male first for night shifts, random otherwise)
                    candidates.sort()
                    
                    # Assign as many as needed
                    for idx in range(min(needed, len(candidates))):
                        _, employee_id = candidates[idx]
                        employee_idx = next(i for i, emp in enumerate(self.employees) if emp[0] == employee_id)
                        
                        assignments.append({
                            'employee_id': employee_id,
                            'employee_idx': employee_idx,
                            'shift': shift,
                            'day': day
                        })
        
        # Second pass: Make sure every worker has at least some shifts
        for i in range(self.num_workers):
            employee_id = self.employees[i][0]
            shifts_count = sum(1 for a in assignments if a['employee_id'] == employee_id)
            
            if shifts_count < 3:
                # Try to assign more shifts
                shifts_needed = 3 - shifts_count
                
                # Try each day and shift combination
                for day in range(self.num_days):
                    for shift in range(self.num_shifts):
                        if shifts_needed <= 0:
                            break
                            
                        if self.check_assignment_feasibility(employee_id, shift, day, assignments):
                            assignments.append({
                                'employee_id': employee_id,
                                'employee_idx': i,
                                'shift': shift,
                                'day': day
                            })
                            shifts_needed -= 1
        
        # Create a schedule from assignments
        worker_schedules = []
        for i in range(self.num_workers):
            employee_id = self.employees_df.iloc[i]['Employee_Id']
            first_name = self.employees_df.iloc[i]['First_Name']
            last_name = self.employees_df.iloc[i]['Last_Name']
            worker_name = f"{first_name} {last_name}"
            
            schedule_row = {'Employee_Id': employee_id, 'WorkerName': worker_name}
            
            for k in range(self.num_days):
                # Find assignment for this employee on this day
                day_assignments = [a for a in assignments if a['employee_idx'] == i and a['day'] == k]
                
                if day_assignments:
                    shift = day_assignments[0]['shift']
                    assigned_shift = self.shift_times[shift]
                else:
                    assigned_shift = 'Off'
                
                schedule_row[day_names[k]] = assigned_shift
            
            worker_schedules.append(schedule_row)
        
        # Calculate shift counts
        shift_counts = {j: [0] * self.num_days for j in range(self.num_shifts)}
        for assignment in assignments:
            shift_counts[assignment['shift']][assignment['day']] += 1
        
        # Prepare shift counts data
        shift_counts_data = []
        for k in range(self.num_days):
            row = {'Day': day_names[k]}
            
            for j in range(self.num_shifts):
                count = shift_counts[j][k]
                min_required = self.shift_constraints.get(day_names[k], {}).get(f'Shift_{j+1}_Min', 0)
                
                # Highlight if below minimum
                label = f'Shift {j+1}'
                if count < min_required:
                    label += f' (MIN: {min_required})'
                
                row[label] = count
            
            shift_counts_data.append(row)
        
        # Export to CSV
        with open('fallback_schedule_solution.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Employee_Id', 'WorkerName'] + day_names)
            writer.writeheader()
            writer.writerows(worker_schedules)
    
            file.write('\n\n')  # Blank line between sections
    
            # Dynamic fieldnames for shift counts including warning labels
            shift_count_fields = ['Day'] + list(shift_counts_data[0].keys())[1:]
            writer = csv.DictWriter(file, fieldnames=shift_count_fields)
            writer.writeheader()
            writer.writerows
# ReRamp
The source code for the ReRamp algorithm, introduced as part of the ECAI 2024 proceedings

The tests were started from Bash with the following command:

```
{ time python ../main.py "$INPUT_NAME" -m -o"$OUTPUT_NAME" -l 2>&1; } &>"$LOG_FILE_NAME"
```

Makespan and sum-of-costs is contained in the log file, the number of robots is from the output file, where plans for each robot are enumerated. The time is measured as the ```real``` component of the ```time``` command.
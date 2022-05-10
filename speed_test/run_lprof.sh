python -m kernprof -l -o speed_test/output/lprof_speedtest.py.lprof speed_test/lprof_speedtest.py
python -m line_profiler speed_test/output/lprof_speedtest.py.lprof
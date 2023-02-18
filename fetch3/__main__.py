# Having this __main__.py file in addition to run.py is a workaround for weirdness with the multiprocessing module
# Multiprocessing module has issues with having everything in the __main__ file


from fetch3.run import main


if __name__ == "__main__":
    main()
# © 2019 University of Illinois Board of Trustees.  All rights reserved.
import sys

class progress:
    def __init__(self, total, step_size, character = "#"):
        self.total         = total;
        self.step_size     = step_size;
        self.__last_update = 0;
        self.character     = character;
        message            = "{0:.2f}".format(0);

        sys.stdout.write(message);
        sys.stdout.flush();

    def __call__(self, current_progress):
        num_steps = int((current_progress / self.total) / self.step_size) - self.__last_update;

        if num_steps > 0:
            message   = self.character * num_steps;
            progress_ = min(current_progress / self.total, 1.0);
            message   = message + " " + "{0:.2f}".format(progress_);

            sys.stdout.write("\b\b\b\b\b");

            sys.stdout.write(message);
            sys.stdout.flush();

            self.__last_update = int((current_progress / self.total) / self.step_size);

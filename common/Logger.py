import io
import sys

class Logger:
    def __init__(self):
        self.lines = []
        self.last_read_idx = 0
        self.clrcount = 0

    def log(self, message, end = '\n'):
        msg_str = str(message) + end

        self.maybe_clear()

        print(msg_str, end = '')
        self.lines.append(msg_str)

    def maybe_clear(self):
        if self.clrcount:
            sys.stdout.write("\033[F\033[K" * self.clrcount) 
            self.clrcount = 0
            sys.stdout.flush()

    def update_last(self, message, end = '\n'):
        msg_str = str(message) + end

        self.maybe_clear()

        sys.stdout.write("\033[F\033[K" * self.lines[-1].count('\n') + msg_str) 
        sys.stdout.flush()

        if self.lines:
            self.lines[-1] = msg_str
        else:
            self.lines.append(msg_str)

    def backtrack(self, count):
        count = min(count, len(self.lines))

        self.maybe_clear()

        #if count <= 0: return

        lines_affected = self.lines[-count:]

        if sys.stdout.isatty():
            self.clrcount = sum(line.count('\n') for line in lines_affected)

        for _ in range(count):
            if self.lines:
                self.lines.pop()

    def read_all(self):
        return "".join(self.lines)

    def clear(self):
        self.lines = []

# shared instance
logger = Logger()

from datetime import datetime
from pytz import timezone


class Logger:
    def __init__(self, filename='log.txt', mode='a+', timezone=''):
        self.file = open(filename, mode)
        self.timezone = timezone
        self.file.write('\n') # Add an empty line
        self.file.flush()
    
    def add_loss(self, loss, epoch=0, title=None, time=0.0):
        if self.timezone == None:
            dt = datetime.now().astimezone(None).strftime('%Y-%m-%d %H:%M:%S%z')
        else:
            dt = datetime.now().astimezone(timezone(self.timezone)).strftime('%Y-%m-%d %H:%M:%S%z')

        template = '[ %s ]' % dt
        if time > 0.0:
            template = '%s [ Took %.2fs ]' % (template, time)
        if title is not None:
            template = '%s [ %s ]' % (template, title)
        template = '%s [ Epoch %03d ]' % (template, epoch)
        template = '%s Loss %.6f' % (template, loss)
        template = '%s \n' % template

        self.file.write(template)
        self.file.flush()
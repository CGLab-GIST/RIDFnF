#  Copyright (c) 2023 CGLab, GIST. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  
#  - Redistributions of source code must retain the above copyright notice, 
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
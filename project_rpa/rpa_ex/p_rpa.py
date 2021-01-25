"""INTEGRATION ENGINE OF RPA FOR PYTHON PACKAGE ~ TEBEL.ORG"""
# Apache License 2.0, Copyright 2020 Tebel.Automation Private Limited
# https://github.com/tebelorg/RPA-Python/blob/master/LICENSE.txt
__author__ = 'Ken Soh <opensource@tebel.org>'
__version__ = '1.27.1'

import subprocess
import os
import sys
import time
import platform

# required for python 2 usage of io.open
if sys.version_info[0] < 3:
    import io


def coord(x_coordinate=0, y_coordinate=0):
    """function to form a coordinate string from x and y integers"""
    return '(' + str(x_coordinate) + ',' + str(y_coordinate) + ')'


def unzip(file_to_unzip=None, unzip_location=None):
    """function to unzip zip file to specified location"""
    import zipfile

    if file_to_unzip is None or file_to_unzip == '':
        print('[RPA][ERROR] - filename missing for unzip()')
        return False
    elif not os.path.isfile(file_to_unzip):
        print('[RPA][ERROR] - file specified missing for unzip()')
        return False

    zip_file = zipfile.ZipFile(file_to_unzip, 'r')

    if unzip_location is None or unzip_location == '':
        zip_file.extractall()
    else:
        zip_file.extractall(unzip_location)

    zip_file.close()
    return True


def echo(text_to_echo=''):
    print(text_to_echo)
    return True


def check(condition_to_check=None, text_if_true='', text_if_false=''):
    if condition_to_check is None:
        print('[RPA][ERROR] - condition missing for check()')
        return False

    if condition_to_check:
        print(text_if_true)

    else:
        print(text_if_false)

    return True


def api(url_to_query=None):
    print('[RPA][INFO] - although TagUI supports calling APIs with headers and body,')
    print('[RPA][INFO] - recommend using requests package with lots of online docs')
    return True


class PRpa(object):

    def __init__(self, robot_name, tagui_root_dir=''):
        self._process = None

        # default timeout in seconds for UI element
        self._tagui_timeout = 10.0

        # default delay in seconds in while loops
        self._tagui_delay = 0.1

        # default debug flag to print debug output
        self._tagui_debug = True

        # flag to track if tagui session is started
        self._tagui_started = False

        # flag to track visual automation connected
        self._tagui_visual = False

        # flag to track chrome browser connected
        self._tagui_chrome = False

        # id to track instruction count from rpa python to tagui
        self._tagui_id = 0

        self._tagui_root_dir = tagui_root_dir
        if platform.system() == 'Windows' and tagui_root_dir == '':
            self._tagui_root_dir = os.environ['APPDATA']

        print(self._tagui_root_dir)

        # to track the original directory when init() was called
        self._robot_name = robot_name
        self._tagui_init_directory = os.path.join(os.getcwd(), robot_name)
        if not os.path.exists(self._tagui_init_directory):
            os.mkdir(self._tagui_init_directory)

        # delete tagui temp output text file to avoid reading old data
        if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python.txt')):
            os.remove(os.path.join(self._tagui_init_directory, 'rpa_python.txt'))

        # define local custom javascript functions for use in tagui
        self._tagui_local_js = \
            """// local custom helper function to check if UI element exists
            // keep checking until timeout is reached before return result
            // effect is interacting with element as soon as it appears
            
            function exist(element_identifier) {
            
                var exist_timeout = Date.now() + casper.options.waitTimeout;
            
                while (Date.now() < exist_timeout) {
                    if (present(element_identifier))
                        return true;
                    else
                       sleep(100);
                }
            
                return false;
            
            }
            
            // function to replace add_concat() in tagui_header.js
            // gain - echoing string with single and double quotes
            // loss - no text-like variables usage since Python env
            
            function add_concat(source_string) {
            
                return source_string;
            
            }
            """

    def _python2_env(self):
        """function to check python version for compatibility handling"""
        if sys.version_info[0] < 3:
            return True
        else:
            return False

    def _python3_env(self):
        """function to check python version for compatibility handling"""
        return not self._python2_env()

    def _py23_decode(self, input_variable=None):
        """function for python 2 and 3 str-byte compatibility handling"""
        if input_variable is None:
            return None
        elif self._python2_env():
            return input_variable
        else:
            return input_variable.decode('utf-8')

    def _py23_encode(self, input_variable=None):
        """function for python 2 and 3 str-byte compatibility handling"""
        if input_variable is None:
            return None
        elif self._python2_env():
            return input_variable
        else:
            return input_variable.encode('utf-8')

    def _py23_open(self, target_filename, target_mode='r'):
        """function for python 2 and 3 open utf-8 compatibility handling"""
        if self._python2_env():
            return io.open(target_filename, target_mode, encoding='utf-8')
        else:
            return open(target_filename, target_mode, encoding='utf-8')

    def _py23_read(self, input_text=None):
        """function for python 2 and 3 read utf-8 compatibility handling"""
        if input_text is None:
            return None
        if self._python2_env():
            return input_text.encode('utf-8')
        else:
            return input_text

    def _py23_write(self, input_text=None):
        """function for python 2 and 3 write utf-8 compatibility handling"""
        if input_text is None:
            return None
        if self._python2_env():
            return input_text.decode('utf-8')
        else:
            return input_text

    def _tagui_read(self):
        """function to read from tagui process live mode interface"""
        # readline instead of read, not expecting user input to tagui
        return self._py23_decode(self._process.stdout.readline())

    def _tagui_write(self, input_text=''):
        """function to write to tagui process live mode interface"""
        # global _process
        self._process.stdin.write(self._py23_encode(input_text))
        self._process.stdin.flush()  # flush to ensure immediate delivery

    def _tagui_output(self):
        """function to wait for tagui output file to read and delete it"""

        # to handle user changing current directory after init() is called
        init_directory_output_file = os.path.join(self._tagui_init_directory, 'rpa_python.txt')
        # sleep to not splurge cpu cycles in while loop
        while not os.path.isfile(init_directory_output_file):
            if os.path.isfile(init_directory_output_file):
                break
            time.sleep(self._tagui_delay)

            # roundabout implementation to ensure backward compatibility
        if os.path.isfile(init_directory_output_file):
            tagui_output_file = self._py23_open(init_directory_output_file, 'r')
            tagui_output_text = self._py23_read(tagui_output_file.read())
            tagui_output_file.close()
            os.remove(init_directory_output_file)
        else:
            tagui_output_file = self._py23_open(init_directory_output_file, 'r')
            tagui_output_text = self._py23_read(tagui_output_file.read())
            tagui_output_file.close()
            os.remove(init_directory_output_file)

        return tagui_output_text

    def _esq(self, input_text=''):
        """function for selective escape of single quote ' for tagui"""
        # [BACKSLASH_QUOTE] marker to work together with send()
        return input_text.replace("'", '[BACKSLASH_QUOTE]')

    def _sdq(self, input_text=''):
        """function to escape ' in xpath for tagui live mode"""
        # change identifier single quote ' to double quote "
        return input_text.replace("'", '"')

    def _started(self):
        return self._tagui_started

    def _visual(self):
        return self._tagui_visual

    def _chrome(self):
        return self._tagui_chrome

    def _python_flow(self):
        """function to create entry tagui flow without visual automation"""
        flow_text = '// NORMAL ENTRY FLOW FOR RPA FOR PYTHON ~ TEBEL.ORG\r\n\r\nlive'
        flow_file = self._py23_open(os.path.join(self._tagui_init_directory, 'rpa_python'), 'w')
        flow_file.write(self._py23_write(flow_text))
        flow_file.close()

    def _visual_flow(self):
        """function to create entry tagui flow with visual automation"""
        flow_text = '// VISUAL ENTRY FLOW FOR RPA FOR PYTHON ~ TEBEL.ORG\r\n' + \
                    '// mouse_xy() - dummy trigger for SikuliX integration\r\n\r\nlive'
        flow_file = self._py23_open(os.path.join(self._tagui_init_directory, 'rpa_python'), 'w')
        flow_file.write(self._py23_write(flow_text))
        flow_file.close()

    def _tagui_local(self):
        """function to create tagui_local.js for custom local functions"""
        javascript_file = self._py23_open(os.path.join(self._tagui_init_directory, 'tagui_local.js'), 'w')
        javascript_file.write(self._py23_write(self._tagui_local_js))
        javascript_file.close()

    def _tagui_delta(self, base_directory=None):
        """function to download stable delta files from tagui cutting edge version"""
        global __version__
        if base_directory is None or base_directory == '':
            return False
        # skip downloading if it is already done before for current release
        if os.path.isfile(base_directory + '/' + 'rpa_python_' + __version__):
            return True

        # define list of key tagui files to be downloaded and synced locally
        delta_list = ['tagui', 'tagui.cmd', 'end_processes', 'end_processes.cmd',
                      'tagui_header.js', 'tagui_parse.php', 'tagui.sikuli/tagui.py.bak']

        for delta_file in delta_list:
            tagui_delta_url = 'https://raw.githubusercontent.com/tebelorg/Tump/master/TagUI-Python/' + delta_file
            tagui_delta_file = base_directory + '/' + 'src' + '/' + delta_file
            if not self.download(tagui_delta_url, tagui_delta_file):
                return False

        # create marker file to skip syncing delta files next time for current release
        delta_done_file = self._py23_open(base_directory + '/' + 'rpa_python_' + __version__, 'w')
        delta_done_file.write(self._py23_write('TagUI installation files used by RPA for Python'))
        delta_done_file.close()
        return True

    def debug(self, on_off=None):
        """function to set debug mode, eg print debug info"""
        if on_off is not None:
            self._tagui_debug = on_off
        return self._tagui_debug

    def init(self, visual_automation=False, chrome_browser=True, userdir='tagui_user_profile'):
        """start and connect to tagui process by checking tagui live mode readiness"""

        if self._tagui_started:
            print('[RPA][ERROR] - use close() before using init() again')
            return False

        # reset id to track instruction count from rpa python to tagui
        self._tagui_id = 0

        # reset variable to track original directory when init() was called

        # to handle user changing current directory after init() is called
        self._tagui_init_directory = os.path.join(os.getcwd(), self._robot_name)
        if not os.path.exists(self._tagui_init_directory):
            os.mkdir(self._tagui_init_directory)

        # # get user home folder location to locate tagui executable
        # if platform.system() == 'Windows':
        #     self._tagui_root_dir = os.environ['APPDATA'] + '/' + 'tagui'

        tagui_executable = self._tagui_root_dir + '/' + 'src' + '/' + 'tagui'
        #  TODO  copy tagui_ex.cmd to tagui path
        # copyfile('./tagui_ex.cmd', self._tagui_root_dir + '/' + 'src' + '/')
        tagui_executable += '_ex.cmd'

        end_processes_executable = self._tagui_root_dir + '/' + 'src' + '/' + 'end_processes'

        # # sync tagui delta files for current release if needed
        # if not self._tagui_delta(self._tagui_root_dir):
        #     return False

        # on Windows, check if there is space in folder path name
        if platform.system() == 'Windows' and ' ' in os.getcwd():
            print('[RPA][INFO] - to use RPA for Python on Windows, avoid space in folder path name')
            return False

        # create entry flow to launch SikuliX accordingly
        if visual_automation:
            # check for working java jdk for visual automation mode
            if platform.system() == 'Windows':
                shell_silencer = '> nul 2>&1'

            # check whether java is installed on the computer
            if os.system('java -version ' + shell_silencer) != 0:
                print('[RPA][INFO] - to use visual automation mode, OpenJDK v8 (64-bit) or later is required')
                print('[RPA][INFO] - download from Amazon Corretto\'s website - https://aws.amazon.com/corretto')
                print('[RPA][INFO] - OpenJDK is preferred over Java JDK which is free for non-commercial use only')
                return False
            else:
                # then check whether it is 64-bit required by sikulix
                os.system('java -version > java_version.txt 2>&1')
                java_version_info = self.load('java_version.txt').lower()
                os.remove('java_version.txt')
                if '64 bit' not in java_version_info and '64-bit' not in java_version_info:
                    print('[RPA][INFO] - to use visual automation mode, OpenJDK v8 (64-bit) or later is required')
                    print('[RPA][INFO] - download from Amazon Corretto\'s website - https://aws.amazon.com/corretto')
                    print('[RPA][INFO] - OpenJDK is preferred over Java JDK which is free for non-commercial use only')
                    return False
                else:
                    # start a dummy first run if never run before, to let sikulix integrate jython
                    sikulix_folder = self._tagui_root_dir + '/' + 'src' + '/' + 'sikulix'
                    if os.path.isfile(sikulix_folder + '/' + 'jython-standalone-2.7.1.jar'):
                        os.system('java -jar ' + sikulix_folder + '/' + 'sikulix.jar -h ' + shell_silencer)
                    self._visual_flow()
        else:
            self._python_flow()

        # create tagui_local.js for custom functions
        self._tagui_local()

        # invoke web browser accordingly with tagui option
        browser_option = ''
        if chrome_browser:
            browser_option = 'chrome'

        # entry shell command to invoke tagui process
        tagui_cmd = tagui_executable + ' ' + self._robot_name + '/rpa_python ' + '-userdir:' + userdir + ' ' + browser_option
        print(tagui_cmd)
        # run tagui end processes script to flush dead processes
        # for eg execution ended with ctrl+c or forget to close()
        # os.system(end_processes_executable)
        try:
            # launch tagui using subprocess
            self._process = subprocess.Popen(
                tagui_cmd, shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # close_fds=True,
            )

            # loop until tagui live mode is ready or tagui process has ended
            while True:

                # failsafe exit if tagui process gets killed for whatever reason
                if self._process.poll() is not None:
                    print('[RPA][ERROR] - following happens when starting TagUI...')
                    print('')
                    os.system(tagui_cmd)
                    print('')
                    self._tagui_visual = False
                    self._tagui_chrome = False
                    self._tagui_started = False
                    return False

                # read next line of output from tagui process live mode interface
                tagui_out = self._tagui_read()
                # check that tagui live mode is ready then start listening for inputs
                if 'LIVE MODE - type done to quit' in tagui_out:
                    # dummy + start line to clear live mode backspace char before listening
                    self._tagui_write('echo "[RPA][' + self._robot_name + '][STARTED]"\n')
                    self._tagui_write(
                        'echo "[RPA][' + self._robot_name + '][' + str(self._tagui_id) + '] - listening for inputs"\n')
                    self._tagui_visual = visual_automation
                    self._tagui_chrome = chrome_browser
                    self._tagui_started = True

                    # loop until tagui live mode is ready and listening for inputs
                    # also check _tagui_started to handle unexpected termination
                    while self._tagui_started and not self._ready():
                        pass
                    if not self._tagui_started:
                        print('[RPA][ERROR] - TagUI process ended unexpectedly')
                        return False

                    # remove generated tagui flow, js code and custom functions files
                    if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python')):
                        os.remove(os.path.join(self._tagui_init_directory, 'rpa_python'))
                    if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python.js')):
                        os.remove(os.path.join(self._tagui_init_directory, 'rpa_python.js'))
                    if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python.raw')):
                        os.remove(os.path.join(self._tagui_init_directory, 'rpa_python.raw'))
                    if os.path.isfile(os.path.join(self._tagui_init_directory, 'tagui_local.js')):
                        os.remove(os.path.join(self._tagui_init_directory, 'tagui_local.js'))

                    # increment id and prepare for next instruction
                    self._tagui_id = self._tagui_id + 1

                    # set variable to track original directory when init() was called
                    # self._tagui_init_directory = os.getcwd()

                    return True

        except Exception as e:
            print('[RPA][ERROR] - ' + str(e))
            self._tagui_visual = False
            self._tagui_chrome = False
            self._tagui_started = False
            return False

    def _ready(self):
        """internal function to check if tagui is ready to receive instructions after init() is called"""
        if not self._tagui_started:
            # print output error in calling parent function instead
            return False

        try:
            # failsafe exit if tagui process gets killed for whatever reason
            if self._process.poll() is not None:
                # print output error in calling parent function instead
                self._tagui_visual = False
                self._tagui_chrome = False
                self._tagui_started = False
                return False

            # read next line of output from tagui process live mode interface
            tagui_out = self._tagui_read()

            # print to screen debug output that is saved to rpa_python.log
            if self.debug():
                sys.stdout.write(tagui_out)
                sys.stdout.flush()

            # check if tagui live mode is listening for inputs and return result
            if tagui_out.strip().startswith('[RPA][') and tagui_out.strip().endswith('] - listening for inputs'):
                return True
            else:
                return False

        except Exception as e:
            print('[RPA][ERROR] - ' + str(e))
            return False

    def send(self, tagui_instruction=None):
        """send next live mode instruction to tagui for processing if tagui is ready"""
        if not self._tagui_started:
            print('[RPA][ERROR] - use init() before using send()')
            return False

        if tagui_instruction is None or tagui_instruction == '':
            return True

        try:
            # failsafe exit if tagui process gets killed for whatever reason
            if self._process.poll() is not None:
                print('[RPA][ERROR] - no active TagUI process to send()')
                self._tagui_visual = False
                self._tagui_chrome = False
                self._tagui_started = False
                return False

            # escape special characters for them to reach tagui correctly
            tagui_instruction = tagui_instruction.replace('\\', '\\\\')
            tagui_instruction = tagui_instruction.replace('\n', '\\n')
            tagui_instruction = tagui_instruction.replace('\r', '\\r')
            tagui_instruction = tagui_instruction.replace('\t', '\\t')
            tagui_instruction = tagui_instruction.replace('\a', '\\a')
            tagui_instruction = tagui_instruction.replace('\b', '\\b')
            tagui_instruction = tagui_instruction.replace('\f', '\\f')

            # special handling for single quote to work with _esq() for tagui
            tagui_instruction = tagui_instruction.replace('[BACKSLASH_QUOTE]', '\\\'')

            # escape backslash to display source string correctly after echoing
            echo_safe_instruction = tagui_instruction.replace('\\', '\\\\')

            # escape double quote because echo step below uses double quotes
            echo_safe_instruction = echo_safe_instruction.replace('"', '\\"')

            # echo live mode instruction, after preparing string to be echo-safe
            self._tagui_write('echo "[RPA][' + str(self._tagui_id) + '] - ' + echo_safe_instruction + '"\n')

            # send live mode instruction to be executed
            self._tagui_write(tagui_instruction + '\n')

            # echo marker text to prepare for next instruction
            self._tagui_write('echo "[RPA][' + str(self._tagui_id) + '] - listening for inputs"\n')

            # loop until tagui live mode is ready and listening for inputs
            # also check _tagui_started to handle unexpected termination
            while self._tagui_started and not self._ready():
                pass
            if not self._tagui_started:
                print('[RPA][ERROR] - TagUI process ended unexpectedly')
                return False

            # increment id and prepare for next instruction
            self._tagui_id = self._tagui_id + 1

            return True

        except Exception as e:
            print('[RPA][ERROR] - ' + str(e))
            return False

    def close(self):
        """disconnect from tagui process by sending 'done' trigger instruction"""

        if not self._tagui_started:
            print('[RPA][ERROR] - use init() before using close()')
            return False

        try:
            # failsafe exit if tagui process gets killed for whatever reason
            if self._process.poll() is not None:
                print('[RPA][ERROR] - no active TagUI process to close()')
                self._tagui_visual = False
                self._tagui_chrome = False
                self._tagui_started = False
                return False

            # send 'done' instruction to terminate live mode and exit tagui
            self._tagui_write('echo "[RPA][FINISHED]"\n')
            self._tagui_write('done\n')

            # loop until tagui process has closed before returning control
            while self._process.poll() is None:
                pass

            # remove again generated tagui flow, js code and custom functions files

            # to handle user changing current directory after init() is called
            if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python')):
                os.remove(os.path.join(self._tagui_init_directory, 'rpa_python'))
            if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python.js')):
                os.remove(os.path.join(self._tagui_init_directory, 'rpa_python.js'))
            if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python.raw')):
                os.remove(os.path.join(self._tagui_init_directory, 'rpa_python.raw'))
            if os.path.isfile(os.path.join(self._tagui_init_directory, 'tagui_local.js')):
                os.remove(os.path.join(self._tagui_init_directory, 'tagui_local.js'))

            # remove generated tagui log and data files if not in debug mode
            if not self.debug():
                # to handle user changing current directory after init() is called
                if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python.log')):
                    os.remove(os.path.join(self._tagui_init_directory, 'rpa_python.log'))
                if os.path.isfile(os.path.join(self._tagui_init_directory, 'rpa_python.txt')):
                    os.remove(os.path.join(self._tagui_init_directory, 'rpa_python.txt'))

            self._tagui_visual = False
            self._tagui_chrome = False
            self._tagui_started = False
            return True

        except Exception as e:
            print('[RPA][ERROR] - ' + str(e))
            self._tagui_visual = False
            self._tagui_chrome = False
            self._tagui_started = False
            return False

    def exist(self, element_identifier=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using exist()')
            return False
        if element_identifier is None or element_identifier == '':
            return False
        # return True for keywords as the computer screen always exists
        if element_identifier.lower() in ['page.png', 'page.bmp']:
            if self._visual():
                return True
            else:
                print('[RPA][ERROR] - page.png / page.bmp requires init(visual_automation = True)')
                return False
        # pre-emptive checks if image files are specified for visual automation
        if element_identifier.lower().endswith('.png') or element_identifier.lower().endswith('.bmp'):
            if not self._visual():
                print('[RPA][ERROR] - ' + element_identifier + ' identifier requires init(visual_automation = True)')
                return False
        # assume that (x,y) coordinates for visual automation always exist
        if element_identifier.startswith('(') and element_identifier.endswith(')'):
            if len(element_identifier.split(',')) in [2, 3]:
                if not any(c.isalpha() for c in element_identifier):
                    if self._visual():
                        return True
                    else:
                        print('[RPA][ERROR] - x, y coordinates require init(visual_automation = True)')
                        return False
        self.send('exist_result = exist(\'' + self._sdq(element_identifier) + '\').toString()')
        self.send('dump exist_result to rpa_python.txt')
        if self._tagui_output() == 'true':
            return True
        else:
            return False

    def url(self, webpage_url=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using url()')
            return False
        if not self._chrome():
            print('[RPA][ERROR] - url() requires init(chrome_browser = True)')
            return False
        if webpage_url is not None and webpage_url != '':
            if webpage_url.startswith('http://') or webpage_url.startswith('https://'):
                if not self.send(self._esq(webpage_url)):
                    return False
                else:
                    return True
            else:
                print('[RPA][ERROR] - URL does not begin with http:// or https:// ')
                return False
        else:
            self.send('dump url() to rpa_python.txt')
            url_result = self._tagui_output()
            return url_result

    def click(self, element_identifier=None, test_coordinate=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using click()')
            return False
        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for click()')
            return False
        if test_coordinate is not None and isinstance(test_coordinate, int):
            element_identifier = coord(element_identifier, test_coordinate)
        if not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False
        elif not self.send('click ' + self._sdq(element_identifier)):
            return False
        else:
            return True

    def rclick(self, element_identifier=None, test_coordinate=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using rclick()')
            return False
        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for rclick()')
            return False
        if test_coordinate is not None and isinstance(test_coordinate, int):
            element_identifier = coord(element_identifier, test_coordinate)
        if not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False
        elif not self.send('rclick ' + self._sdq(element_identifier)):
            return False
        else:
            return True

    def dclick(self, element_identifier=None, test_coordinate=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using dclick()')
            return False
        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for dclick()')
            return False
        if test_coordinate is not None and isinstance(test_coordinate, int):
            element_identifier = coord(element_identifier, test_coordinate)
        if not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False
        elif not self.send('dclick ' + self._sdq(element_identifier)):
            return False
        else:
            return True

    def hover(self, element_identifier=None, test_coordinate=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using hover()')
            return False
        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for hover()')
            return False
        if test_coordinate is not None and isinstance(test_coordinate, int):
            element_identifier = coord(element_identifier, test_coordinate)
        if not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False
        elif not self.send('hover ' + self._sdq(element_identifier)):
            return False
        else:
            return True

    def type(self, element_identifier=None, text_to_type=None, test_coordinate=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using type()')
            return False
        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for type()')
            return False
        if text_to_type is None or text_to_type == '':
            print('[RPA][ERROR] - text missing for type()')
            return False
        if test_coordinate is not None and isinstance(text_to_type, int):
            element_identifier = coord(element_identifier, text_to_type)
            text_to_type = test_coordinate
        if not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False
        elif not self.send('type ' + self._sdq(element_identifier) + ' as ' + self._esq(text_to_type)):
            return False
        else:
            return True

    def select(self, element_identifier=None, option_value=None, test_coordinate1=None, test_coordinate2=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using select()')
            return False
        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for select()')
            return False
        if option_value is None or option_value == '':
            print('[RPA][ERROR] - option value missing for select()')
            return False
        if element_identifier.lower() in ['page.png', 'page.bmp'] or option_value.lower() in ['page.png', 'page.bmp']:
            print('[RPA][ERROR] - page.png / page.bmp identifiers invalid for select()')
            return False
        if test_coordinate1 is not None and test_coordinate2 is not None and \
                isinstance(option_value, int) and isinstance(test_coordinate2, int):
            element_identifier = coord(element_identifier, option_value)
            option_value = coord(test_coordinate1, test_coordinate2)
            # pre-emptive checks if image files are specified for visual automation
        if element_identifier.lower().endswith('.png') or element_identifier.lower().endswith('.bmp'):
            if not self._visual():
                print('[RPA][ERROR] - ' + element_identifier + ' identifier requires init(visual_automation = True)')
                return False
        if option_value.lower().endswith('.png') or option_value.lower().endswith('.bmp'):
            if not self._visual():
                print('[RPA][ERROR] - ' + option_value + ' identifier requires init(visual_automation = True)')
                return False
        if not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False
        elif not self.send('select ' + self._sdq(element_identifier) + ' as ' + self._esq(option_value)):
            return False
        else:
            return True

    def read(self, element_identifier=None, test_coordinate1=None, test_coordinate2=None, test_coordinate3=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using read()')
            return ''

        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for read()')
            return ''

        if test_coordinate1 is not None and isinstance(test_coordinate1, int):
            if test_coordinate2 is not None and isinstance(test_coordinate2, int):
                if test_coordinate3 is not None and isinstance(test_coordinate3, int):
                    element_identifier = coord(element_identifier, test_coordinate1) + '-'
                    element_identifier = element_identifier + coord(test_coordinate2, test_coordinate3)

        if element_identifier.lower() != 'page' and not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return ''

        else:
            self.send('read ' + self._sdq(element_identifier) + ' to read_result')
            self.send('dump read_result to rpa_python.txt')
            read_result = self._tagui_output()
            return read_result

    def snap(self, element_identifier=None, filename_to_save=None, test_coord1=None, test_coord2=None,
             test_coord3=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using snap()')
            return False

        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for snap()')
            return False

        if filename_to_save is None or filename_to_save == '':
            print('[RPA][ERROR] - filename missing for snap()')
            return False

        if test_coord2 is not None and test_coord3 is None:
            print('[RPA][ERROR] - filename missing for snap()')
            return False

        if isinstance(element_identifier, int) and isinstance(filename_to_save, int):
            if test_coord1 is not None and isinstance(test_coord1, int):
                if test_coord2 is not None and isinstance(test_coord2, int):
                    if test_coord3 is not None and test_coord3 != '':
                        element_identifier = coord(element_identifier, filename_to_save) + '-'
                        element_identifier = element_identifier + coord(test_coord1, test_coord2)
                        filename_to_save = test_coord3

        if element_identifier.lower() != 'page' and not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False

        elif not self.send('snap ' + self._sdq(element_identifier) + ' to ' + self._esq(filename_to_save)):
            return False

        else:
            return True

    def load(self, filename_to_load=None):
        if filename_to_load is None or filename_to_load == '':
            print('[RPA][ERROR] - filename missing for load()')
            return ''

        elif not os.path.isfile(filename_to_load):
            print('[RPA][ERROR] - cannot load file ' + filename_to_load)
            return ''

        else:
            load_input_file = self._py23_open(filename_to_load, 'r')
            load_input_file_text = self._py23_read(load_input_file.read())
            load_input_file.close()
            return load_input_file_text

    def dump(self, text_to_dump=None, filename_to_save=None):
        if text_to_dump is None:
            print('[RPA][ERROR] - text missing for dump()')
            return False

        elif filename_to_save is None or filename_to_save == '':
            print('[RPA][ERROR] - filename missing for dump()')
            return False

        else:
            dump_output_file = self._py23_open(filename_to_save, 'w')
            dump_output_file.write(self._py23_write(text_to_dump))
            dump_output_file.close()
            return True

    def write(self, text_to_write=None, filename_to_save=None):
        if text_to_write is None:
            print('[RPA][ERROR] - text missing for write()')
            return False

        elif filename_to_save is None or filename_to_save == '':
            print('[RPA][ERROR] - filename missing for write()')
            return False

        else:
            write_output_file = self._py23_open(filename_to_save, 'a')
            write_output_file.write(self._py23_write(text_to_write))
            write_output_file.close()
            return True

    def ask(self, text_to_prompt=''):
        if self._chrome():
            return self.dom("return prompt('" + self._esq(text_to_prompt) + "')")

        else:
            if text_to_prompt == '':
                space_padding = ''
            else:
                space_padding = ' '

            if self._python2_env():
                return raw_input(text_to_prompt + space_padding)
            else:
                return input(text_to_prompt + space_padding)

    def keyboard(self, keys_and_modifiers=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using keyboard()')
            return False
        if keys_and_modifiers is None or keys_and_modifiers == '':
            print('[RPA][ERROR] - keys to type missing for keyboard()')
            return False
        if not self._visual():
            print('[RPA][ERROR] - keyboard() requires init(visual_automation = True)')
            return False
        elif not self.send('keyboard ' + self._esq(keys_and_modifiers)):
            return False
        else:
            return True

    def mouse(self, mouse_action=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using mouse()')
            return False
        if mouse_action is None or mouse_action == '':
            print('[RPA][ERROR] - \'down\' / \'up\' missing for mouse()')
            return False
        if not self._visual():
            print('[RPA][ERROR] - mouse() requires init(visual_automation = True)')
            return False
        elif mouse_action.lower() != 'down' and mouse_action.lower() != 'up':
            print('[RPA][ERROR] - \'down\' / \'up\' missing for mouse()')
            return False
        elif not self.send('mouse ' + mouse_action):
            return False
        else:
            return True

    def table(self, element_identifier=None, filename_to_save=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using table()')
            return False

        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for table()')
            return False

        elif filename_to_save is None or filename_to_save == '':
            print('[RPA][ERROR] - filename missing for table()')
            return False

        elif not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False

        elif not self.send('table ' + self._sdq(element_identifier) + ' to ' + self._esq(filename_to_save)):
            return False

        else:
            return True

    def upload(self, element_identifier=None, filename_to_upload=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using upload()')
            return False

        if element_identifier is None or element_identifier == '':
            print('[RPA][ERROR] - target missing for upload()')
            return False

        elif filename_to_upload is None or filename_to_upload == '':
            print('[RPA][ERROR] - filename missing for upload()')
            return False

        elif not self.exist(element_identifier):
            print('[RPA][ERROR] - cannot find ' + element_identifier)
            return False

        elif not self.send('upload ' + self._sdq(element_identifier) + ' as ' + self._esq(filename_to_upload)):
            return False

        else:
            return True

    def download(self, download_url=None, filename_to_save=None):
        """function for python 2/3 compatible file download from url"""

        if download_url is None or download_url == '':
            print('[RPA][ERROR] - download URL missing for download()')
            return False

        # if not given, use last part of url as filename to save
        if filename_to_save is None or filename_to_save == '':
            download_url_tokens = download_url.split('/')
            filename_to_save = download_url_tokens[-1]

        # delete existing file if exist to ensure freshness
        if os.path.isfile(filename_to_save):
            os.remove(filename_to_save)

        # handle case where url is invalid or has no content
        try:
            if self._python2_env():
                import urllib
                urllib.urlretrieve(download_url, filename_to_save)
            else:
                import urllib.request
                urllib.request.urlretrieve(download_url, filename_to_save)

        except Exception as e:
            print('[RPA][ERROR] - failed downloading from ' + download_url + '...')
            print(str(e))
            return False

        # take the existence of downloaded file as success
        if os.path.isfile(filename_to_save):
            return True

        else:
            print('[RPA][ERROR] - failed downloading to ' + filename_to_save)
            return False

    def frame(self, main_frame=None, sub_frame=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using frame()')
            return False

        if not self._chrome():
            print('[RPA][ERROR] - frame() requires init(chrome_browser = True)')
            return False

        # reset webpage context to document root, by sending custom tagui javascript code
        self.send('js chrome_step("Runtime.evaluate", {expression: "mainframe_context = null"})')
        self.send('js chrome_step("Runtime.evaluate", {expression: "subframe_context = null"})')
        self.send('js chrome_context = "document"; frame_step_offset_x = 0; frame_step_offset_y = 0;')

        # return True if no parameter, after resetting webpage context above
        if main_frame is None or main_frame == '':
            return True

        # set webpage context to main frame specified, by sending custom tagui javascript code
        frame_identifier = '(//frame|//iframe)[@name="' + main_frame + '" or @id="' + main_frame + '"]'
        if not self.exist(frame_identifier):
            print('[RPA][ERROR] - cannot find frame with @name or @id as \'' + main_frame + '\'')
            return False

        self.send('js new_context = "mainframe_context"')
        self.send('js frame_xpath = \'(//frame|//iframe)[@name="' + main_frame + '" or @id="' + main_frame + '"]\'')
        self.send('js frame_rect = chrome.getRect(xps666(frame_xpath))')
        self.send('js frame_step_offset_x = frame_rect.left; frame_step_offset_y = frame_rect.top;')
        self.send(
            'js chrome_step("Runtime.evaluate", {expression: new_context + " = document.evaluate(\'" + frame_xpath + "\'," + chrome_context + ",null,XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,null).snapshotItem(0).contentDocument"})')
        self.send('js chrome_context = new_context')

        # set webpage context to sub frame if specified, by sending custom tagui javascript code
        if sub_frame is not None and sub_frame != '':
            frame_identifier = '(//frame|//iframe)[@name="' + sub_frame + '" or @id="' + sub_frame + '"]'
            if not self.exist(frame_identifier):
                print('[RPA][ERROR] - cannot find sub frame with @name or @id as \'' + sub_frame + '\'')
                return False

            self.send('js new_context = "subframe_context"')
            self.send('js frame_xpath = \'(//frame|//iframe)[@name="' + sub_frame + '" or @id="' + sub_frame + '"]\'')
            self.send('js frame_rect = chrome.getRect(xps666(frame_xpath))')
            self.send('js frame_step_offset_x = frame_rect.left; frame_step_offset_y = frame_rect.top;')
            self.send(
                'js chrome_step("Runtime.evaluate", {expression: new_context + " = document.evaluate(\'" + frame_xpath + "\'," + chrome_context + ",null,XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,null).snapshotItem(0).contentDocument"})')
            self.send('js chrome_context = new_context')

        return True

    def popup(self, string_in_url=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using popup()')
            return False

        if not self._chrome():
            print('[RPA][ERROR] - popup() requires init(chrome_browser = True)')
            return False

        # reset webpage context to main page, by sending custom tagui javascript code
        self.send(
            'js if (chrome_targetid !== "") {found_targetid = chrome_targetid; chrome_targetid = ""; chrome_step("Target.detachFromTarget", {sessionId: found_targetid});}')

        # return True if no parameter, after resetting webpage context above
        if string_in_url is None or string_in_url == '':
            return True

        # set webpage context to the popup tab specified, by sending custom tagui javascript code
        self.send('js found_targetid = ""; chrome_targets = []; ws_message = chrome_step("Target.getTargets", {});')
        self.send(
            'js try {ws_json = JSON.parse(ws_message); if (ws_json.result.targetInfos) chrome_targets = ws_json.result.targetInfos; else chrome_targets = [];} catch (e) {chrome_targets = [];}')
        self.send(
            'js chrome_targets.forEach(function(target) {if (target.url.indexOf("' + string_in_url + '") !== -1) found_targetid = target.targetId;})')
        self.send(
            'js if (found_targetid !== "") {ws_message = chrome_step("Target.attachToTarget", {targetId: found_targetid}); try {ws_json = JSON.parse(ws_message); if (ws_json.result.sessionId !== "") found_targetid = ws_json.result.sessionId; else found_targetid = "";} catch (e) {found_targetid = "";}}')
        self.send('js chrome_targetid = found_targetid')

        # check if chrome_targetid is successfully set to sessionid of popup tab
        self.send('dump chrome_targetid to rpa_python.txt')
        popup_result = self._tagui_output()
        if popup_result != '':
            return True
        else:
            print('[RPA][ERROR] - cannot find popup tab containing URL string \'' + string_in_url + '\'')
            return False

    def run(self, command_to_run=None):
        if command_to_run is None or command_to_run == '':
            print('[RPA][ERROR] - command(s) missing for run()')
            return ''

        else:
            if platform.system() == 'Windows':
                command_delimiter = ' & '
            else:
                command_delimiter = '; '
            return self._py23_decode(subprocess.check_output(
                command_to_run + command_delimiter + 'exit 0',
                stderr=subprocess.STDOUT,
                shell=True))

    def dom(self, statement_to_run=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using dom()')
            return ''

        if statement_to_run is None or statement_to_run == '':
            print('[RPA][ERROR] - statement(s) missing for dom()')
            return ''

        if not self._chrome():
            print('[RPA][ERROR] - dom() requires init(chrome_browser = True)')
            return ''

        else:
            self.send('dom ' + statement_to_run)
            self.send('dump dom_result to rpa_python.txt')
            dom_result = self._tagui_output()
            return dom_result

    def vision(self, command_to_run=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using vision()')
            return False

        if command_to_run is None or command_to_run == '':
            print('[RPA][ERROR] - command(s) missing for vision()')
            return False

        if not self._visual():
            print('[RPA][ERROR] - vision() requires init(visual_automation = True)')
            return False

        elif not self.send('vision ' + command_to_run):
            return False

        else:
            return True

    def timeout(self, timeout_in_seconds=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using timeout()')
            return False

        # global _tagui_timeout

        if timeout_in_seconds is None:
            return float(self._tagui_timeout)

        else:
            _tagui_timeout = float(timeout_in_seconds)

        if not self.send('timeout ' + str(timeout_in_seconds)):
            return False

        else:
            return True

    def present(self, element_identifier=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using present()')
            return False

        if element_identifier is None or element_identifier == '':
            return False

        # return True for keywords as the computer screen is always present
        if element_identifier.lower() in ['page.png', 'page.bmp']:
            if self._visual():
                return True
            else:
                print('[RPA][ERROR] - page.png / page.bmp requires init(visual_automation = True)')
                return False

        # pre-emptive checks if image files are specified for visual automation
        if element_identifier.lower().endswith('.png') or element_identifier.lower().endswith('.bmp'):
            if not self._visual():
                print('[RPA][ERROR] - ' + element_identifier + ' identifier requires init(visual_automation = True)')
                return False

        # assume that (x,y) coordinates for visual automation always exist
        if element_identifier.startswith('(') and element_identifier.endswith(')'):
            if len(element_identifier.split(',')) in [2, 3]:
                if not any(c.isalpha() for c in element_identifier):
                    if self._visual():
                        return True
                    else:
                        print('[RPA][ERROR] - x, y coordinates require init(visual_automation = True)')
                        return False

        self.send('present_result = present(\'' + self._sdq(element_identifier) + '\').toString()')
        self.send('dump present_result to rpa_python.txt')
        if self._tagui_output() == 'true':
            return True
        else:
            return False

    def count(self, element_identifier=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using count()')
            return int(0)

        if element_identifier is None or element_identifier == '':
            return int(0)

        if not self._chrome():
            print('[RPA][ERROR] - count() requires init(chrome_browser = True)')
            return int(0)

        self.send('count_result = count(\'' + self._sdq(element_identifier) + '\').toString()')
        self.send('dump count_result to rpa_python.txt')
        return int(self._tagui_output())

    def title(self):
        if not self._started():
            print('[RPA][ERROR] - use init() before using title()')
            return ''

        if not self._chrome():
            print('[RPA][ERROR] - title() requires init(chrome_browser = True)')
            return ''

        self.send('dump title() to rpa_python.txt')
        title_result = self._tagui_output()
        return title_result

    def text(self):
        if not self._started():
            print('[RPA][ERROR] - use init() before using text()')
            return ''

        if not self._chrome():
            print('[RPA][ERROR] - text() requires init(chrome_browser = True)')
            return ''

        self.send('dump text() to rpa_python.txt')
        text_result = self._tagui_output()
        return text_result

    def timer(self):
        if not self._started():
            print('[RPA][ERROR] - use init() before using timer()')
            return float(0)

        self.send('dump timer() to rpa_python.txt')
        timer_result = self._tagui_output()
        return float(timer_result)

    def mouse_xy(self):
        if not self._started():
            print('[RPA][ERROR] - use init() before using mouse_xy()')
            return ''

        if not self._visual():
            print('[RPA][ERROR] - mouse_xy() requires init(visual_automation = True)')
            return ''

        self.send('dump mouse_xy() to rpa_python.txt')
        mouse_xy_result = self._tagui_output()
        return mouse_xy_result

    def mouse_x(self):
        if not self._started():
            print('[RPA][ERROR] - use init() before using mouse_x()')
            return int(0)

        if not self._visual():
            print('[RPA][ERROR] - mouse_x() requires init(visual_automation = True)')
            return int(0)

        self.send('dump mouse_x() to rpa_python.txt')
        mouse_x_result = self._tagui_output()
        return int(mouse_x_result)

    def mouse_y(self):
        if not self._started():
            print('[RPA][ERROR] - use init() before using mouse_y()')
            return int(0)

        if not self._visual():
            print('[RPA][ERROR] - mouse_y() requires init(visual_automation = True)')
            return int(0)

        self.send('dump mouse_y() to rpa_python.txt')
        mouse_y_result = self._tagui_output()
        return int(mouse_y_result)

    def clipboard(self, text_to_put=None):
        if not self._started():
            print('[RPA][ERROR] - use init() before using clipboard()')
            return False

        if not self._visual():
            print('[RPA][ERROR] - clipboard() requires init(visual_automation = True)')
            return False

        if text_to_put is None:
            self.send('dump clipboard() to rpa_python.txt')
            clipboard_result = self._tagui_output()
            return clipboard_result

        elif not self.send("js clipboard('" + text_to_put.replace("'", '[BACKSLASH_QUOTE]') + "')"):
            return False

        else:
            return True

    def wait(self, delay_in_seconds=5.0):
        time.sleep(float(delay_in_seconds))
        return True

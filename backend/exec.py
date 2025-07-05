import sys
import traceback
from io import StringIO
from flask import jsonify, request, Flask
from flask_cors import CORS
from multiprocessing import Process, Queue
import queue
app = Flask(__name__)
CORS(app)
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

@app.route('/exec', methods=['POST', 'GET'])
def execute():
    code = request.data.decode('utf-8')  # 确保解码为字符串
    output_queue = Queue()
    p = Process(target=_run_code, args=(code, output_queue))
    p.start()
    try:
        output, error, stack_trace = output_queue.get(timeout=30)
    except queue.Empty:
        output, error, stack_trace = "", "执行超时（超过 30 秒）考虑通信send-receive不匹配导致的阻塞，或其中一方代码出错", ""
    finally:
        p.terminate()

    response = {
        "output": output,
        "error": error,
        "stack_trace": stack_trace if error else ""
    }
    return jsonify(response)

def _run_code(code, output_queue):
    original_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = Tee(buffer, original_stdout)

    error = None
    stack_trace = None
    try:
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, {})
    except Exception as e:
        error = str(e)
        # 获取完整的堆栈跟踪
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    finally:
        sys.stdout = original_stdout
        output_queue.put((buffer.getvalue(), error, stack_trace))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="解释器服务")
    parser.add_argument("--port", type=int, help="端口", required=True)
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port)

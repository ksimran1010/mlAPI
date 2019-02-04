from flask import Flask, redirect, request, jsonify

app = Flask(__name__)

def startServer(prt):
    @app.route('/read', methods=['POST'])
    def readFromPage():
        newReq = request.get_json()

        if newReq['method'] == 'logistic':

            from logReg import logistic
            obj = logistic(newReq['iterations'])
            acc = obj.run()
            print ('final accuracy by logistic regression is ', acc)

        if newReq['method'] == 'linear':

            from linReg import LetsGo
            obj = LetsGo(newReq['iterations'])
            obj.run()

        print(newReq, type(newReq))
        return jsonify(newReq)

    # @app.route("/print",methods = ['GET'])
    # def printToPage():
    #     return 'iam a string'
    #

    app.run(port=prt)


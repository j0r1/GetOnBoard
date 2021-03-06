#! /usr/bin/env python

from SimpleHTTPServer import SimpleHTTPRequestHandler, test

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)

if __name__ == '__main__':
    test(HandlerClass=CORSHTTPRequestHandler)

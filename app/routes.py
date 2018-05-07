# -*- coding: utf-8 -*-
from flask import Flask, jsonify
from app import app
import recommendation
@app.route('/')

@app.route('/todo/rec/<keyword>', methods=['GET'])
def get_movie(keyword):
    rec=recommendation.get_recommendations(keyword)
    return jsonify('Recommended Movies:{}'.format(rec))


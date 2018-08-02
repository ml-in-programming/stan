import sys
import os
sys.path.append(os.getcwd() + '/backend/')

from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from project_predictor import run_backend, get_status

import json


@csrf_exempt
def get_name(request):
    if request.method == 'GET':
        address = request.GET.get('address', '')
        # if address != '':
        #     run_backend(address)
    else:
        address = ''

    return render(request, 'request_page.html', {'address': address})


def load_content(request):
    result = 'Invalid request'
    if request.method == 'POST':
        address = request.POST.get('address')
        counts = request.POST.get('count')
        if address != '':
            try:
                result = run_backend(address, int(counts))
            except ValueError as err:
                result = json.dumps({'error': str(err)})

    return HttpResponse(result)


def request_status(request):
    status = 'Invalid request'
    if request.method == 'POST':
        address = request.POST.get('address')
        if address != '':
            status = get_status(address)
    return HttpResponse(json.dumps({'status': status}))


def check_health(request):
    return HttpResponse(status=200)

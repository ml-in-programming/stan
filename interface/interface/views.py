import sys
import os
sys.path.append(os.getcwd() + '/backend/')

from django.http import HttpResponse
from django.template.loader import render_to_string
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from project_predictor import run_backend

import git
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
            except git.GitCommandError:
                result = json.dumps({'error': 'Invalid repository address'})
            except ValueError as err:
                result = json.dumps({'error': str(err)})

    return HttpResponse(result)


def check_health(request):
    return HttpResponse(status=200)

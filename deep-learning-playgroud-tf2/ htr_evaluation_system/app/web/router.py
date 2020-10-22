from flask import request, jsonify, render_template, redirect, url_for

from . import router_blue_print
from ..model.models import *

common_api_headers = {
    'content-type': 'application/json'
}


@router_blue_print.route('/')
def index():
    page = 1
    limit = 20
    result = THtrDatum.query.filter(THtrDatum.pred_result == 'G').filter_by(is_verificated=0).limit(limit).offset(
        (page - 1) * limit).all()
    j = []
    for item in result:
        i = item.serialize()
        # print(item, i)
        j.append(i)

    return render_template('index.html', results=j)


# 分页查询未校正的选项图片列表
@router_blue_print.route('/regulation_list', methods={"GET"})
def get_regulation_list():
    page = int(request.args.get("page"))
    limit = int(request.args.get("limit"))
    result = THtrDatum.query.filter_by(is_verificated=0).limit(limit).offset((page - 1) * limit).all()
    j = []
    for item in result:
        i = item.serialize()
        # print(item, i)
        j.append(i)
    return jsonify(j), 200, common_api_headers


# 提交校正后的图片结果
@router_blue_print.route('/regulate', methods={"POST"})
def regulate():
    params = request.form
    for p, v in params.items():
        if v is None or v == '':
            print('delete', p, str(v))
            result = THtrDatum.query.filter_by(id=int(p)).first()
            db.session.delete(result)
            db.session.commit()
        else:
            print('update', p, str(v))
            THtrDatum.query.filter_by(id=int(p)).update(dict(real_result=str(v).upper(), is_verificated=1))
            db.session.commit()

    return redirect(url_for('web.index'))

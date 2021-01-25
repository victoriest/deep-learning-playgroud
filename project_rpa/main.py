import subprocess
import time

import rpa_ex as rpaa
import os

from rpa_ex.p_rpa import PRpa
from multiprocessing import Process, Pool


def run_rpa(robot_name, url):
    rpa1 = PRpa(robot_name=robot_name)

    rpa1.init(userdir=robot_name)
    rpa1.timeout(1)
    rpa1.url(url)
    rpa1.close()


def run_test(robot_name, tagui_root):
    rpaa = PRpa(robot_name=robot_name, tagui_root_dir=tagui_root)
    rpaa.init(userdir=robot_name)
    rpaa.timeout(1)

    rpaa.url(r"http://github.com")
    rpaa.wait(2)
    # login 如果没有登录, 则登录. 并在登录过程中提示询问密码
    if rpaa.exist(r"/html/body/div[1]/header/div/div[2]/div[2]/a[1]"):
        r = rpaa.click(r"/html/body/div[1]/header/div/div[2]/div[2]/a[1]")
        rpaa.type('//*[@id="login_field"]', rpaa.ask('root:'))
        rpaa.type('//*[@id="password"]', rpaa.ask('密码:'))
        rpaa.click('//*[@id="login"]/div[4]/form/input[14]')
        rpaa.wait(2)

    # 跳转到profile页面
    rpaa.click(r"/html/body/div[1]/header/div[7]/details/summary")
    rpaa.click(r'/html/body/div[1]/header/div[7]/details/details-menu/a[8]')
    rpaa.wait(2)

    # 打开并上传头像, 这里, #avatar_upload是通过查看网页源码得到的
    rpaa.click('//*[@id="js-pjax-container"]/div/div/div[2]/div[2]/div[2]/dl/dd/div/details/summary/img')
    rpaa.upload(r'#avatar_upload', r'D:\_onedrive\OneDrive\图片\pic\icon.jpg')
    rpaa.wait(5)
    # 确认头像显示区域
    rpaa.click(r'/html/body/details/details-dialog/form/div[2]/button')
    rpaa.wait(5)

    # logout 退出登录
    rpaa.click(r"/html/body/div[1]/header/div[7]/details/summary")
    rpaa.click(r'/html/body/div[1]/header/div[7]/details/details-menu/form/button')
    rpaa.wait(2)
    rpaa.close()


if __name__ == '__main__':
    # rpaa.init(userdir='est1')
    # rpaa.timeout(1)
    # rpaa.url(r"http://github.com")
    # rpaa.wait(2)
    # # login 如果没有登录, 则登录. 并在登录过程中提示询问密码
    # if rpaa.exist(r"/html/body/div[1]/header/div/div[2]/div[2]/a[1]"):
    #     r = rpaa.click(r"/html/body/div[1]/header/div/div[2]/div[2]/a[1]")
    #     rpaa.type('//*[@id="login_field"]', 'victoriest')
    #     rpaa.type('//*[@id="password"]', rpaa.ask('密码:'))
    #     rpaa.click('//*[@id="login"]/div[4]/form/input[14]')
    #     rpaa.wait(2)
    #
    # # 跳转到profile页面
    # rpaa.click(r"/html/body/div[1]/header/div[7]/details/summary")
    # rpaa.click(r'/html/body/div[1]/header/div[7]/details/details-menu/a[8]')
    # rpaa.wait(2)
    #
    # # 打开并上传头像, 这里, #avatar_upload是通过查看网页源码得到的
    # rpaa.click('//*[@id="js-pjax-container"]/div/div/div[2]/div[2]/div[2]/dl/dd/div/details/summary/img')
    # rpaa.upload(r'#avatar_upload', r'D:\_onedrive\OneDrive\图片\pic\icon.jpg')
    # rpaa.wait(5)
    # # 确认头像显示区域
    # rpaa.click(r'/html/body/details/details-dialog/form/div[2]/button')
    # rpaa.wait(5)
    #
    # # logout 退出登录
    # rpaa.click(r"/html/body/div[1]/header/div[7]/details/summary")
    # rpaa.click(r'/html/body/div[1]/header/div[7]/details/details-menu/form/button')
    # rpaa.wait(2)
    # rpaa.close()

    p1 = Process(target=run_test, args=('est1', 'D:/robot_tagui'))
    p2 = Process(target=run_test, args=('est2', 'D:/robot_tagui_2'))

    p1.start()
    time.sleep(10)
    p2.start()

    p1.join()
    p2.join()

    # p = Pool(2)
    # p.apply_async(run_test, args=('est1', ))
    # p.apply_async(run_rpa, args=('est2', 'http://bilibili.com'))
    #
    # p.close()
    # p.join()

    # run_test('est1', 'D:/robot_tagui')
    # run_test('est2', 'D:/robot_tagui_2')

    # rpa1 = PRpa(robot_name='est1', tagui_root_dir='D:/robot_tagui')
    # rpa1.init(userdir='est1')
    # rpa1.timeout(1)
    # rpa1.url('http://bilibili.com')
    #
    # rpa2 = PRpa(robot_name='est2', tagui_root_dir='D:/robot_tagui_2')
    # rpa2.init(userdir='est2')
    # rpa2.timeout(1)
    # rpa2.url('http://baidu.com')
    #
    # rpa1.close()
    # rpa2.close()



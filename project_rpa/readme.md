可以通过魔改tagui.cmd中的配置实现多用户多脚本(line986):
`start "" "%chrome_command%" --user-data-dir="%~dp0chrome\tagui_user_profile_c1" !chrome_switches! !window_size! !headless_switch!`

rpa-python可以通过修改tagui.cmd文件, 将 用户dir作为参数 传给脚本, 然后修改tagui.py, 在init的时候传用户dir参数

自定义了tagui.cmd文件  加入 -userdir: 参数, 可以自定义chrome的--user-data-dir 参数


如果希望启动多个多个实例, 并行运行多个任务: 
* 要运行多少实例, 就需要复制多少个tgui应用程序文件夹副本 
* 开启多个cmd 
* 命令行下分别运行多个脚本: `[tagui路径]/src/tagui_ex.cmd [tag脚本路径] -userdir:[user-data名]`

需要注意的一点: 
* 多cmd窗口启动脚本需要间隔些时间, 保险起见当脚本启动浏览器后, 打开网页内容后, 再启动下一个脚本. 


基于rpa-python, 定制:
* 需要将, rpa-python启动时, 调用的tagui. 拷贝出来, 放到你的目标路径. 
*  实例化PRpa对象: PRpa(robot_name=[本实例tagui名字], tagui_root_dir=[上述tagui根目录])
*  需要启动多少个PRpa对象, 就要复制出几个tagui程序目录
*  不能并行实例化PRpa对象, 必须要等到一个PRpa对象初始化完毕, 并打开第一个网址后(即, 调用rpa.url方法), 才能进行下一个PRpa对象的实例化. 

Ref:
* <https://zyhh.me/windows/chrome-multi.html>
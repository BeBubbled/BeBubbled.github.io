# Ubuntu的初始配置

[TOC]

## 进入ubuntu引导盘

nvidia显卡出现乱码卡主无法进入到安装界面解决方法:

先关机,再次进入到选择try ubuntu/install ubuntu界面

选择try ubuntu/install ubuntu,按键盘e编辑修改启动文件,

在

```
linux   /vmlinuz-XXXXXX root=UUID=XXXX-XXXX-XXXX-XXXXXXXXX ro quiet splash $vt_handoff
```

的splash与\$vt_handoff之间添加一段nomodeset,经过处理,你看到的文字应该是:

```
linux   /vmlinuz-XXXXXX root=UUID=XXXX-XXXX-XXXX-XXXXXXXXX ro quiet splash nomodeset $vt_handoff
```

之后即可正常进入try ubuntu/install ubuntu

注:在完成所有ubuntu安装后,若之前曾建出现需要添加nomodeset才能正常安装,则第一次启动ubuntu仍然需要按e编辑进行如上操作

## 分区问题:

我所预备的空闲分区大小: 330GB ,具体分配多少请自行决定,后期可分区工具修改

分区问题:

| 挂载点   | 大小       | 新分区的类型 | 新分区的位置 | 用于         |
| ----- | -------- | ------ | ------ | ---------- |
| /boot | 400MB    | 主分区    | 空间起始位置 | Ext4日志文件系统 |
| swap  | 16384MB  | 逻辑分区   | 空间起始位置 | 交换空间       |
| /     | 167936MB | 逻辑分区   | 空间起始位置 | Ext4日志文件系统 |
| /home | 152576   | 逻辑分区   | 空间起始位置 | Ext4日志文件系统 |

安装启动引导器的设备，选择/boot所在分区

## 首次启动ubuntu

### 修改分辨率

打开Software&Updates,安装nvidia驱动

![image-20200413140108423](../../Downloads/image-20200413140108423.png)

### 安装homebrew

**注意: 如果你尽享要下载cmake等文件队员吗进行编译,重构,那么homebrew中的环境变量可能导致当场爆炸. 慎用我的环境变量参考,已爆炸一次**

官方参考文档的echo $PATH....错误

请按照如下方案(如果缺少依赖,terminal会以文字告诉你,请复制terminal反馈给你的dependencies 命令自行安装)(参考自: ):

```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
echo 'export PATH="/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin/:$PATH"' >>~/.bashrc
echo 'export MANPATH="/home/linuxbrew/.linuxbrew/share/man:$MANPATH"' >>~/.bashrc
echo 'export INFOPATH="/home/linuxbrew/.linuxbrew/share/info:$INFOPATH"' >>~/.bashrc
```

这样配置环境变量存在环境变量污染问题,请自行斟酌,也可考虑如下链接解决

[知乎解决方案](https://zhuanlan.zhihu.com/p/81840844)

### 中文输入法问题

打开ubuntu software搜索fcitx并安装

依次打开settings, Region& Language, Manage Installed Languages,

在Keyboard input method system选择fcitx,

继续此界面点击Apply System-Wide, 勾选Chinese Simplify

[搜狗输入法](https://pinyin.sogou.com/linux/?r=pinyin)安装,并关闭搜狗一切快捷键,半角符号

#### 永久将cpaslock替换为Lctrl

```
cd /usr/share/X11/xkb/symbols/
sudo chmod 777 pc
gedit pc
```

紧接着你会看到如下窗口

![image-20200413141840688](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20200413141840688.png)

从第一行往下数(不包含空行)第15行,你会看到

```
key <CAPS> {    [ Control_L        ]    };
```

Caps_Lock替换为Control_L

为保险起见,将原有的第十五行前方添加//变为注释,也即你会看到

![image-20200413142044211](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20200413142044211.png)

保存退出GUI(可视化)窗口

terminal继续输入如下命令

```
sudo chmod 755 pc
```

相关注意事项: 此方法实现的键盘映射无法触发快捷键. 

原因分析: 发现时及时更改成了Caps_Lock+ctrl 这个组合键,需要进一步改进

pc这个文件中的其他内容请不要随便更改, 我试过了, 然后重启,内核崩溃. 这个bug还未提交至ubuntu官方,主要是懒

#### Cap_Lock设置为输入法切换键

打开fcitx configuration-Global Config,Trigger Input Method点击第一个按钮按下Caps_lock,你会发现已经替换为Lctrl

![image-20200413142434352](https://raw.githubusercontent.com/BeBubbled/PicGoImages-WorkSpace/master/image-20200413142434352.png)

## 软件安装

[Jetbrains全家桶](https://www.jetbrains.com/toolbox-app/)

Gnome GUI控制界面: tweak

## 插件/美化

[Gnome shell插件](https://extensions.gnome.org/)

[Gnome美化资源](https://www.gnome-look.org/)

可供参考美化教程: [Ubuntu 18.04美化（朝着Mac的方向整容）](https://zhuanlan.zhihu.com/p/67693607)

# 写在最后

$$
\Huge{\textcolor{red}{\text{WSLg yyds !!!!!!!}}}
$$

[WSLg install instructions](https://github.com/microsoft/wslg)

CASE

XCAS

asd


[分区参考]:https://www.jianshu.com/p/70da2204e24d
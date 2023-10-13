# vim configure

**MUST READ**

add `export COLORTERM=truecolor` to Windows terminal setting / WSL Ubuntu .bashrc file!

# Tmux trige bashrc

see [new tmux sessions do not source bashrc](https://unix.stackexchange.com/questions/320465/new-tmux-sessions-do-not-source-bashrc-file) for details.

the fastest way is to put "source ~/.bashrc" into oneof ~/.bash_profile, ~/.bash_login, and ~/.profile is ok.

* to make vim color theme work under tmux, set environments:

export TERM="screen-256color"

* install zenburn color scheme

download this file: https://github.com/jnurmine/Zenburn/blob/master/colors/zenburn.vim to `~/.vim/colors`

* install for coc

see https://zhuanlan.zhihu.com/p/145793963 for more details.

see https://nodejs.org/en/download/package-manager#debian-and-ubuntu-based-linux-distributions for install nodejs on ubuntu.

* install vim-plug

run this commond in terminal:

''' shell 
$ curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
'''

* add custom plugins witin following code block

''' vim
call plug#begin('~/.vim/plugged')
Plug 'itchyny/lightline.vim'
...
call plug#end()
'''

* some vim-plug commands

  - :PlugStatus
    用于检查插安装状态
  - :PlugInstall
    用于安装插件
  - :PlugUpdate
    用于更新插件
  - 删除插件
    首先将`.vimrc`中要删除的插件行使用`"`注释掉或者直接删除，然后使用 `:PlugClean` 命令即可将vim配置文件中所有未声明的插件卸载。
  - 审查插件
    有时，更新插件可能有的插件引入 bug 或者无法正常工作，要解决这个问题，你可以简单地回滚有问题的插件。输入 :PlugDiff 命令，然后按回车键查看上次 :PlugUpdate的更改，并在每个段落上按 X 将每个插件回滚到更新前的前一个状态。
  - 升级 vim-plug
    要升级vim-plug本身，请输入：
    
    ''' vimL
    :PlugUpgrade
    '''

    参考文件： [Vim-plug: 极简Vim插件管理器](https://zhuanlan.zhihu.com/p/38156442)

* some shotcut

  - ctrl-p: find filename using Leaderf
  - gd: goto defined places
  - gr: goto all referenced places
  - gy: goto the type definition place
  - :AsyncRun + commond: run the command in the quickfix window
  - :bp goto previous buffer
  - :bn goto next buffer

* 一个看上去比较有用的博客：https://zhuanlan.zhihu.com/p/599417117


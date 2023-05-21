## TODO
- [ ] 测试Q-learning agent表现
   - [x] 测试不同训练次数下的表现
   - [ ] 可视化
- [ ] 实现近似Q-Learning
- [x] 实现search agent
- [x] 使创建函数可控制taxi初始位置，乘客与目的地位置
- [ ] Markov agent
   - [ ] 天气
   - [x] 战争迷雾
   - [x] 使创建函数可控制taxi初始位置，乘客与目的地位置

## taxi

[taxi tutorial](https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/)

### openai gym
[openai gym](https://gymnasium.farama.org/)

#### API & ENV
[environment](https://gymnasium.farama.org/api/env/)

#### Wrappers(additional part)
[warppers](https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper) 

---
## 一些教程
### say goodbye to ppt
1. Markdown Preview Enhanced [官方中文文档](https://shd101wyy.github.io/markdown-preview-enhanced/#/zh-cn/presentation?id=presentation-front-matter)
   在vscode插件商店下载*Markdown Preview Enhanced* 
2. 右键md文档 `MPE: 打开侧边预览` 来浏览slides
3. 在幻灯片预览界面右键可以在浏览器中打开
4. ~~uninstall microsoft ppt~~

### 开始使用Git

[runoob教程](https://www.runoob.com/git/git-tutorial.html) 

1. 本地需要下载git 

   > See https://git-scm.com/book/en/v2/Getting-Started-Installing-Git. Get git installed on your system. 

   1.1 如果是Windows，可以考虑添加到path：google it

2. 创建本地仓库

   在目标文件夹下打开git命令行，输入`git init` 

3. 添加远程仓库（"<",">"不用输）

   `git remote add <远程名称，一般是origin> <url,从网页上复制过来>` 

   更多关于远程仓库以及添加sshkey来避免输入密码：[runoob](https://www.runoob.com/git/git-remote-repo.html) 

4. 从远程仓库拉取：更新其他人的代码到本地`git pull <远程主机名> <远程分支名>:<本地分支名>` 

5. 上传本地代码

   1. `git add <要更新的文件名>`, 可以用`git add .`全部提交
   2. `git commit -m"更新信息"` 
   3. `git push <远程主机名> <远程分支名>`(`git push origin main`) 

   这三部是分开的，可以分别add/commit 多次再 commit/push

6. 新建分支 [懒得写了](https://www.runoob.com/git/git-branch.html) 

**Tips**: 你还是可以在vscode中安装插件来解决一切问题，赞美vscode




---

###### other

[project idea 整合网站](https://www.gocoder.one/blog/reinforcement-learning-project-ideas/#3-simulate-control-tasks-with-pybullet) 


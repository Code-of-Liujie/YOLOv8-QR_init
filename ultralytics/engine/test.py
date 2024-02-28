import pip

# 获取已安装的包列表
installed_packages = pip.get_installed_distributions()

# 输出每个包的名称和版本号
for package in installed_packages:
    print(package.key, package.version)
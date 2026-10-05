# 作者手动 Zenodo 存档与 DOI 回填

本项目不自动操作你的 Zenodo，也不提前编造 DOI。当前 `CITATION.cff`
仅含论文源稿已有的作者、ORCID 与机构信息，请你最终核对。

1. 在 GitHub 核对仓库文件、许可证、README 与成功运行的检查记录；明确
   存档的是代码和数值复现材料，不是未经授权的规划原件或 GISCO 边界。
2. 你登录 Zenodo，在 GitHub 集成中启用此仓库，然后为冻结提交创建正式
   GitHub Release（拟用 `v1.0.0`）；亦可选择 Zenodo 的手动文件上传路线。
   本仓库不替你启用集成或触发 DOI。注意集成启用时间与 Release 的关系。
3. 检查 Zenodo 的作者、版本、资源类型、文件列表与许可证。软件使用 MIT；
   作者的分析数据使用 CC BY 4.0，但第三方来源的特定条款仍适用。
   不要在页面上声称所有上游材料均已获 MIT/CC BY 转授权。
4. 存档完成后，核对 DOI 确实可解析，下载存档文件重新验证校验和，并
   确认其对应本次冻结的 tag/commit，而不是后续变动的默认分支。
5. 将实际版本 DOI、仓库 URL、tag 和 commit 提供给助手。论文的方法引用
   应指向所用版本 DOI；如同时提供总版本 DOI，清楚区分二者。
6. 回填 `CITATION.cff` 的真实 `doi` / `repository-code` 信息，以及论文
   Data Availability Statement 和参考文献。回填造成的新提交不应被
   误说成已包含在旧的归档版本里；必要时发布新的修订存档版本。

投稿声明只描述真实状态。未完成 Zenodo 时，可以写代码和数据已在实际
GitHub URL 公开，并说明尚无归档 DOI；不能提前写“已在 Zenodo 存档”。
不要使用“本研究未生成新数据”——本研究确实生成了分析数据和证书。

官方说明（2026-10-05 核验）：

- [Zenodo：从 GitHub 归档版本](https://help.zenodo.org/docs/github/archive-software/github-upload/)
- [Zenodo：CITATION.cff](https://help.zenodo.org/docs/github/describe-software/citation-file/)
- [Land 作者指南](https://www.mdpi.com/journal/land/instructions)
- [MDPI 研究数据和 AI 披露政策](https://www.mdpi.com/ethics)

只使用 `CITATION.cff`，暂不添加重复的 `.zenodo.json`；Zenodo 官方说明
两者共存时只读取 `.zenodo.json`，容易发生元数据不一致。DOI 不是代码
正确性认证；科学主张仍必须由证据与可复算结果支撑。

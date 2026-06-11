import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: '我的网站',
  description: '前端开发者的文档、博客与学习笔记',

  // 顶部导航菜单
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '博客', link: '/blog/' },
      { text: '学习', link: '/learn/' },
      { text: '关于', link: '/about' }
    ],

    // 侧边栏：按板块配置
    sidebar: {
      '/blog/': [
        {
          text: '博客',
          items: [
            { text: '全部文章', link: '/blog/' }
          ]
        },
        {
          text: '前端',
          collapsed: false,
          items: [
            { text: '示例：第一篇前端文章', link: '/blog/frontend/example' }
          ]
        },
        {
          text: '随笔',
          collapsed: false,
          items: [
            { text: '示例：一篇随笔', link: '/blog/notes/example' }
          ]
        }
      ],
      '/learn/': [
        {
          text: '学习',
          items: [
            { text: '总览', link: '/learn/' }
          ]
        },
        {
          text: '视频剪辑',
          collapsed: false,
          items: [
            { text: '示例：剪辑入门笔记', link: '/learn/video/example' }
          ]
        },
        {
          text: '英语',
          collapsed: false,
          items: [
            { text: '示例：英语学习笔记', link: '/learn/english/example' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/' }
    ],

    footer: {
      message: '基于 VitePress 构建',
      copyright: 'Copyright © 2026'
    },

    // 中文化
    outline: { label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新' },
    returnToTopLabel: '返回顶部',
    darkModeSwitchLabel: '主题',
    sidebarMenuLabel: '菜单'
  }
})

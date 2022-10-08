# docsify Cheatsheet

## Useful command

```shell
docsify serve BeBubbled.github.io
```

## Tips

#### Q: Avoid copy paste file to post folder?

```shell
ln -s original_fil_path new_linked_path
```

Current [typora](https://typora.io/) not suoport symlink without -s on MacOS

#### Q: what if space character in filename/path

use

```shell
%20
```

 replace the space in <u>citation place/Hyperlink</u>

## Backup

[Awesome Docsify](https://github.com/docsifyjs/awesome-docsify)

folder structure:

* XXXXX.github.io
  * _navbar.md
  * _sidebar.md
  * .git
  * .nojekyll
  * index.html
  * README.md
  * symlink_of_the_article

index.html setting:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Document</title>
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="description" content="Description">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, minimum-scale=1.0">
  <!-- Theme: Simple -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/docsify-themeable@0/dist/css/theme-simple.css">


  <style>
    :root {
      /* Reduce the font size */
      /*--base-font-size: 14px;*/

      /* Change the theme color hue */
      /*--theme-hue: 325;*/

      /* Change the sidebar bullets */
      /*--sidebar-nav-link-before-content: '😀';*/
    }
  </style>


</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      name: 'BeBubble',
      repo: '',
      //load side bar
      loadSidebar: true,
      loadSidebar: '_sidebar.md',
      subMaxLevel: 5,

      //no emoji
      noEmoji: true,

      //load navigation bar
      loadNavbar: true,
      loadNavbar: '_navbar.md',

      //scroll to the top of the screen when the route is changed
      auto2top: true,

      toc: {
        tocMaxLevel: 5,
        target: 'h1, h2, h3, h4, h5'
      },


      //docsify-copy-code
      copyCode: {
        buttonText : 'Copy to clipboard',
        errorText  : 'Error',
        successText: 'Copied'
      },



      mergeNavbar: true,
      themeable: {
          readyTransition : true, // default
          responsiveTables: true  // default
      }
    }
  </script>
  <!-- Docsify v4 -->
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
  <!-- docsify-themeable (latest v0.x.x) -->
  <script src="https://cdn.jsdelivr.net/npm/docsify-themeable@0/dist/js/docsify-themeable.min.js"></script>
  <!-- toc plugin https://github.com/justintien/docsify-plugin-toc -->
  <script src="//unpkg.com/docsify-plugin-toc"></script>

  <!-- CDN files for docsify-katex -->
  <script src="//cdn.jsdelivr.net/npm/docsify-katex@latest/dist/docsify-katex.js"></script>
  <!-- or <script src="//cdn.jsdelivr.net/gh/upupming/docsify-katex@latest/dist/docsify-katex.js"></script> -->
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/katex@latest/dist/katex.min.css"/>

  <!-- Latest v2.x.x -->
  <!-- https://github.com/jperasmus/docsify-copy-code -->
  <script src="https://unpkg.com/docsify-copy-code@2"></script>





</body>
</html>
```

{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :special-members: __contains__,__getitem__,__iter__,__len__,__add__,__sub__,__mul__,__div__,__neg__,__hash__

   .. rubric:: Methods
   .. autosummary::

   {% for item in methods %}
   {%- if item != "__init__" %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}

   

.. include:: {{fullname}}.examples

.. raw:: html

    <div style='clear:both'></div>
